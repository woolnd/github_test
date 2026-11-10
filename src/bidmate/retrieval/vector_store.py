import time

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from bidmate import config

load_dotenv()

_vector_stores: dict[str, Chroma] = {}
_MAX_ADD_ATTEMPTS = 3
_QUERY_MAX_ATTEMPTS = 2
_RRF_K = 60


def get_vector_store(
    embeddings: Embeddings | None = None,
    collection_name: str | None = None,
) -> Chroma:
    name = collection_name or config.COLLECTION_NAME
    if name in _vector_stores and embeddings is None:
        return _vector_stores[name]

    embedding_fn = embeddings or OpenAIEmbeddings(model=config.EMBEDDING_MODEL_NAME)
    store = Chroma(
        collection_name=name,
        embedding_function=embedding_fn,
        persist_directory=str(config.CHROMA_FOLDER),
    )
    if embeddings is None:
        _vector_stores[name] = store
    return store


def _add_with_retry(store: Chroma, batch: list[Document], ids: list[str]) -> None:
    for attempt in range(1, _MAX_ADD_ATTEMPTS + 1):
        try:
            store.add_documents(batch, ids=ids)
            return
        except Exception:
            if attempt == _MAX_ADD_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))


def _validate_chunks(chunks: list[Document]) -> None:
    for i, doc in enumerate(chunks):
        if "chunk_id" not in doc.metadata:
            raise ValueError(f"{i}번째 조각에 chunk_id가 없음 — page_content 앞부분: {doc.page_content[:30]!r}")


def save_chunks(
    chunks: list[Document],
    batch_size: int = 100,
    collection_name: str | None = None,
) -> None:
    if not chunks:
        print("저장할 조각이 없어 건너뜀")
        return
    _validate_chunks(chunks)

    store = get_vector_store(collection_name=collection_name)
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        ids = [doc.metadata["chunk_id"] for doc in batch]
        try:
            _add_with_retry(store, batch, ids)
        except Exception:
            print(f"[저장 실패] {start}번째 조각부터 막힘 — chunks[{start}:]만 다시 넘길 것")
            raise
        print(f"[저장] {min(start + batch_size, total)}/{total}건 완료")
    print(f"저장 끝 — 총 {total}건, 컬렉션 '{collection_name or config.COLLECTION_NAME}'")


def _apply_distance_cutoff(results: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
    if config.MAX_DISTANCE is None:
        return results
    return [pair for pair in results if pair[1] <= config.MAX_DISTANCE]


def _search_with_retry(store: Chroma, question: str, k: int, filter_dict: dict | None) -> list[tuple[Document, float]]:
    for attempt in range(1, _QUERY_MAX_ATTEMPTS + 1):
        try:
            return store.similarity_search_with_score(question, k=k, filter=filter_dict)
        except Exception:
            if attempt == _QUERY_MAX_ATTEMPTS:
                raise
            time.sleep(0.5)
    return []


def search_chunks(
    question: str,
    project_ids: list[str],
    k: int = 5,
    collection_name: str | None = None,
) -> list[tuple[Document, float]]:
    if k <= 0:
        raise ValueError(f"k는 1 이상이어야 함 (받은 값: {k})")
    store = get_vector_store(collection_name=collection_name)
    filter_dict = {"project_id": {"$in": project_ids}} if project_ids else None
    results = _search_with_retry(store, question, k, filter_dict)
    results = sorted(results, key=lambda pair: pair[1])
    return _apply_distance_cutoff(results)


def search_chunks_diverse(
    question: str,
    project_ids: list[str],
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    collection_name: str | None = None,
) -> list[Document]:
    if k <= 0:
        raise ValueError(f"k는 1 이상이어야 함 (받은 값: {k})")
    store = get_vector_store(collection_name=collection_name)
    filter_dict = {"project_id": {"$in": project_ids}} if project_ids else None
    return store.max_marginal_relevance_search(
        question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter_dict
    )


def search_chunks_hybrid(
    question: str,
    project_ids: list[str],
    k: int = 5,
    fetch_k: int = 20,
    collection_name: str | None = None,
) -> list[tuple[Document, float]]:
    if k <= 0:
        raise ValueError(f"k는 1 이상이어야 함 (받은 값: {k})")
    store = get_vector_store(collection_name=collection_name)
    filter_dict = {"project_id": {"$in": project_ids}} if project_ids else None
    dense_results = _search_with_retry(store, question, fetch_k, filter_dict)
    if not dense_results:
        return []
    dense_results = sorted(dense_results, key=lambda pair: pair[1])

    docs = [doc for doc, _score in dense_results]
    tokenized_corpus = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(question.split())
    bm25_order = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)

    rrf_scores = [0.0] * len(docs)
    for rank in range(len(docs)):
        rrf_scores[rank] += 1 / (_RRF_K + rank + 1)
    for rrf_rank, doc_index in enumerate(bm25_order):
        rrf_scores[doc_index] += 1 / (_RRF_K + rrf_rank + 1)

    ranked = sorted(range(len(docs)), key=lambda i: rrf_scores[i], reverse=True)
    return [(docs[i], rrf_scores[i]) for i in ranked[:k]]
