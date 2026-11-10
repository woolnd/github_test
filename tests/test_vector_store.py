import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from bidmate.retrieval.vector_store import (
    get_vector_store,
    save_chunks,
    search_chunks,
    search_chunks_hybrid,
)


def _chunk(chunk_id: str, project_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "project_id": project_id,
            "agency_name": "가짜기관",
            "project_name": "가짜사업",
        },
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    from bidmate import config
    monkeypatch.setattr(config, "CHROMA_FOLDER", tmp_path / "chroma_test")
    return get_vector_store(embeddings=DeterministicFakeEmbedding(size=8), collection_name="test_col")


def test_search_chunks_rejects_non_positive_k():
    with pytest.raises(ValueError):
        search_chunks("질문", [], k=0)


def test_filter_only_returns_matching_project(store):
    store.add_documents(
        [_chunk("doc1-0001", "doc1", "내용 A"), _chunk("doc2-0001", "doc2", "내용 B")],
        ids=["doc1-0001", "doc2-0001"],
    )
    results = store.similarity_search_with_score("질문", k=5, filter={"project_id": {"$in": ["doc1"]}})
    assert all(doc.metadata["project_id"] == "doc1" for doc, _score in results)


def test_empty_filter_means_whole_collection(store):
    store.add_documents(
        [_chunk("doc1-0001", "doc1", "내용 A"), _chunk("doc2-0001", "doc2", "내용 B")],
        ids=["doc1-0001", "doc2-0001"],
    )
    results = store.similarity_search_with_score("질문", k=5, filter=None)
    assert len(results) == 2


def test_duplicate_chunk_id_does_not_duplicate(store):
    chunk = _chunk("doc1-0001", "doc1", "같은 조각")
    store.add_documents([chunk], ids=["doc1-0001"])
    store.add_documents([chunk], ids=["doc1-0001"])
    assert len(store.get()["ids"]) == 1


def test_results_sorted_by_distance_ascending(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    monkeypatch.setitem(vs._vector_stores, "test_col", store)
    store.add_documents(
        [_chunk(f"doc1-{i:04d}", "doc1", f"조각 {i} 내용 {i * 13}") for i in range(10)],
        ids=[f"doc1-{i:04d}" for i in range(10)],
    )
    results = search_chunks("조각 5 내용", ["doc1"], k=10, collection_name="test_col")
    distances = [score for _, score in results]
    assert distances == sorted(distances)


def test_max_distance_cutoff_filters_far_results(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs
    from bidmate import config

    monkeypatch.setitem(vs._vector_stores, "test_col", store)
    store.add_documents(
        [_chunk(f"doc1-{i:04d}", "doc1", f"조각 {i} 내용 {i * 13}") for i in range(10)],
        ids=[f"doc1-{i:04d}" for i in range(10)],
    )
    unfiltered = search_chunks("조각 5 내용", ["doc1"], k=10, collection_name="test_col")
    cutoff = unfiltered[len(unfiltered) // 2][1]

    monkeypatch.setattr(config, "MAX_DISTANCE", cutoff)
    filtered = search_chunks("조각 5 내용", ["doc1"], k=10, collection_name="test_col")
    assert all(score <= cutoff for _, score in filtered)
    assert len(filtered) < len(unfiltered)


def test_hybrid_search_rejects_non_positive_k():
    with pytest.raises(ValueError):
        search_chunks_hybrid("질문", [], k=0)


def test_hybrid_search_boosts_keyword_match(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    monkeypatch.setitem(vs._vector_stores, "test_col", store)
    store.add_documents(
        [
            _chunk("doc1-0001", "doc1", "예산은 1억 5천만원이며 2024년 5월에 마감된다"),
            _chunk("doc1-0002", "doc1", "이러닝 시스템 구축 사업의 요구사항은 사용자 인증이다"),
            _chunk("doc1-0003", "doc1", "포털 고도화 사업은 학내 커뮤니티 기능을 포함한다"),
        ],
        ids=["doc1-0001", "doc1-0002", "doc1-0003"],
    )
    results = search_chunks_hybrid("예산은 얼마야", ["doc1"], k=3, collection_name="test_col")
    rrf_by_id = {doc.metadata["chunk_id"]: score for doc, score in results}
    assert rrf_by_id["doc1-0001"] > rrf_by_id["doc1-0003"]


def test_hybrid_search_respects_k(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    monkeypatch.setitem(vs._vector_stores, "test_col", store)
    store.add_documents(
        [_chunk(f"doc1-{i:04d}", "doc1", f"조각 {i} 내용 예산 {i * 13}") for i in range(10)],
        ids=[f"doc1-{i:04d}" for i in range(10)],
    )
    results = search_chunks_hybrid("예산", ["doc1"], k=3, collection_name="test_col")
    assert len(results) == 3


class _FlakySearchStore:
    def __init__(self, real_store, fail_times):
        self._real_store = real_store
        self._fail_times = fail_times
        self.calls = 0

    def similarity_search_with_score(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ConnectionError("일시적 네트워크 에러")
        return self._real_store.similarity_search_with_score(*args, **kwargs)


def test_search_chunks_retries_once_on_transient_error(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    store.add_documents([_chunk("doc1-0001", "doc1", "내용")], ids=["doc1-0001"])
    flaky = _FlakySearchStore(store, fail_times=1)
    monkeypatch.setitem(vs._vector_stores, "test_col", flaky)

    results = search_chunks("질문", [], k=1, collection_name="test_col")
    assert flaky.calls == 2
    assert len(results) == 1


def test_search_chunks_raises_after_exhausting_retries(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    flaky = _FlakySearchStore(store, fail_times=99)
    monkeypatch.setitem(vs._vector_stores, "test_col", flaky)

    with pytest.raises(ConnectionError):
        search_chunks("질문", [], k=1, collection_name="test_col")
    assert flaky.calls == 2


def test_save_chunks_rejects_chunk_missing_chunk_id(store, monkeypatch):
    import bidmate.retrieval.vector_store as vs

    monkeypatch.setitem(vs._vector_stores, "test_col", store)
    broken = Document(page_content="내용", metadata={"project_id": "doc1"})
    with pytest.raises(ValueError):
        save_chunks([broken], collection_name="test_col")
    assert len(store.get()["ids"]) == 0
