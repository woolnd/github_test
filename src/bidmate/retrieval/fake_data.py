from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class FakeProjectInfo:
    project_id: str
    agency_name: str
    project_name: str
    budget: int
    deadline: str


FAKE_PROJECTS = [
    FakeProjectInfo("가나다공단_이러닝.hwp", "가나다공단", "통합 이러닝 플랫폼 구축", 150_000_000, "2024-05-31"),
    FakeProjectInfo("국민연금공단_포털.pdf", "국민연금공단", "이러닝시스템 구축", 220_000_000, "2024-06-15"),
    FakeProjectInfo("고려대_포털.hwp", "고려대학교", "학내 포털 고도화", 90_000_000, "2024-04-20"),
    FakeProjectInfo("광주과기원_학사.pdf", "광주과학기술원", "학사정보시스템 구축", 130_000_000, "2024-07-10"),
]


def _make_chunk(project: FakeProjectInfo, order: int, body: str) -> Document:
    header = f"[{project.agency_name} / {project.project_name}]"
    suffix = "info" if order == 0 else f"{order:04d}"
    return Document(
        page_content=f"{header}\n{body}",
        metadata={
            "chunk_id": f"{project.project_id}-{suffix}",
            "project_id": project.project_id,
            "agency_name": project.agency_name,
            "project_name": project.project_name,
        },
    )


def make_fake_chunks() -> list[Document]:
    chunks: list[Document] = []
    for project in FAKE_PROJECTS:
        chunks.append(_make_chunk(
            project, 0,
            f"기본 정보: 발주 기관 {project.agency_name} | 예산 {project.budget:,}원 | 입찰 마감일 {project.deadline}",
        ))
        chunks.append(_make_chunk(project, 1, "3. 사업 요구사항 이러닝 시스템 고도화 및 사용자 인증 모듈 구축을 포함한다."))
        chunks.append(_make_chunk(project, 2, "4. 제출 방식 제안서는 전자 제출 시스템을 통해 접수한다."))
    return chunks
