from dataclasses import dataclass

from bidmate.retrieval.project_finder import find_project_ids


@dataclass
class _FakeProject:
    project_id: str
    agency_name: str
    project_name: str


_PROJECTS = [
    _FakeProject("doc1", "국민연금공단", "이러닝시스템 구축"),
    _FakeProject("doc2", "고려대학교", "포털 고도화"),
]


def test_finds_project_by_agency_name():
    assert find_project_ids("국민연금공단 예산이 얼마야?", _PROJECTS) == ["doc1"]


def test_finds_project_by_project_name():
    assert find_project_ids("포털 고도화 사업 요구사항은?", _PROJECTS) == ["doc2"]


def test_finds_multiple_projects_for_comparison_question():
    found = find_project_ids("국민연금공단과 고려대학교 사업을 비교해줘", _PROJECTS)
    assert set(found) == {"doc1", "doc2"}


def test_returns_empty_list_when_no_match():
    assert find_project_ids("전혀 관련 없는 질문입니다", _PROJECTS) == []


def test_matches_fullwidth_parenthesis_variant():
    projects = [_FakeProject("doc3", "(주)가나다공단", "이러닝 플랫폼 구축")]
    assert find_project_ids("（주）가나다공단 예산은?", projects) == ["doc3"]
