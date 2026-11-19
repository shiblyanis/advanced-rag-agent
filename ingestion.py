from pathlib import Path
from typing import Iterable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def read_upload(filename: str, content: bytes) -> list[dict]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        import io
        reader = PdfReader(io.BytesIO(content))
        return [{"text": page.extract_text() or "", "page": i + 1} for i, page in enumerate(reader.pages)]
    if suffix in {".txt", ".md", ".markdown"}:
        return [{"text": content.decode("utf-8", errors="ignore"), "page": None}]
    raise ValueError("Supported formats: PDF, TXT, Markdown")


def make_parent_child_chunks(pages: Iterable[dict], filename: str, child_size: int = 200, parent_size: int = 1000) -> tuple[list[dict], list[dict]]:
    parents, children = [], []
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_size, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_size, chunk_overlap=40)
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        for pidx, parent in enumerate(parent_splitter.split_text(text)):
            parent_id = f"{filename}:{page.get('page')}:{pidx}"
            parents.append({"id": parent_id, "text": parent, "filename": filename, "page": page.get("page")})
            for cidx, child in enumerate(child_splitter.split_text(parent)):
                children.append({"id": f"{parent_id}:{cidx}", "parent_id": parent_id, "text": child, "filename": filename, "page": page.get("page")})
    return parents, children
