import unicodedata


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    return " ".join(normalized.split())


def find_project_ids(question: str, projects: list) -> list[str]:
    question = _normalize_text(question)
    found: list[str] = []
    for project in projects:
        agency = _normalize_text(project.agency_name)
        name = _normalize_text(project.project_name)
        if (agency and agency in question) or (name and name in question):
            found.append(project.project_id)
    return found
