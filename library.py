import os
import re
import shutil
from typing import Any, Optional

from langchain_community.vectorstores import FAISS

from embedding import build_index, extend_index, index_size, load_index
from extract import extract_many, extract_pages, summarize_pages

DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
INDEX_DIR = os.path.join(DATA_DIR, "index")

_UNSAFE = re.compile(r"[^A-Za-z0-9._ ()\[\]&,+-]")


def safe_name(filename: str) -> str:
    name = _UNSAFE.sub("_", os.path.basename(filename or "")).strip()
    if not name.lower().endswith(".pdf"):
        name = f"{name or 'document'}.pdf"
    return name


class DocumentLibrary:
    def __init__(self, upload_dir: str = UPLOAD_DIR, index_dir: str = INDEX_DIR):
        self.upload_dir = upload_dir
        self.index_dir = index_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        self._index: Optional[FAISS] = None
        self._loaded = False

    @property
    def index(self) -> Optional[FAISS]:
        if not self._loaded:
            self._index = load_index(self.index_dir)
            self._loaded = True
        return self._index

    def files(self) -> list[str]:
        if not os.path.isdir(self.upload_dir):
            return []
        return sorted(
            os.path.join(self.upload_dir, f)
            for f in os.listdir(self.upload_dir)
            if f.lower().endswith(".pdf")
        )

    def documents(self) -> list[dict[str, Any]]:
        entries = []
        for path in self.files():
            try:
                stats = summarize_pages(extract_pages(path))["documents"]
            except Exception:
                stats = []
            info = stats[0] if stats else {
                "filename": os.path.basename(path),
                "title": os.path.basename(path),
                "total_pages": 0,
                "indexed_pages": 0,
                "words": 0,
            }
            info["size_bytes"] = os.path.getsize(path)
            entries.append(info)
        return entries

    def pages_for(self, filename: str) -> list[dict[str, Any]]:
        path = os.path.join(self.upload_dir, safe_name(filename))
        if not os.path.exists(path):
            return []
        return extract_pages(path)

    def store_upload(self, filename: str, stream) -> str:
        name = safe_name(filename)
        path = os.path.join(self.upload_dir, name)
        stem, suffix = os.path.splitext(name)
        counter = 1
        while os.path.exists(path):
            path = os.path.join(self.upload_dir, f"{stem} ({counter}){suffix}")
            counter += 1
        with open(path, "wb") as target:
            shutil.copyfileobj(stream, target)
        return path

    def add(self, paths: list[str]) -> dict[str, Any]:
        pages = extract_many(paths)
        if not pages:
            raise ValueError("No text could be extracted. The PDFs may be scanned images.")
        self._index = extend_index(self.index, pages, self.index_dir)
        self._loaded = True
        return summarize_pages(pages)

    def rebuild(self) -> int:
        shutil.rmtree(self.index_dir, ignore_errors=True)
        self._index = None
        self._loaded = True

        paths = self.files()
        if not paths:
            return 0

        pages = extract_many(paths)
        if not pages:
            return 0

        self._index = build_index(pages, self.index_dir)
        return len(pages)

    def remove(self, filename: str) -> bool:
        path = os.path.join(self.upload_dir, safe_name(filename))
        if not os.path.exists(path):
            return False
        os.remove(path)
        self.rebuild()
        return True

    def reset(self) -> None:
        shutil.rmtree(self.index_dir, ignore_errors=True)
        shutil.rmtree(self.upload_dir, ignore_errors=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        self._index = None
        self._loaded = True

    def stats(self) -> dict[str, Any]:
        documents = self.documents()
        return {
            "documents": len(documents),
            "pages": sum(d["total_pages"] for d in documents),
            "words": sum(d["words"] for d in documents),
            "chunks": index_size(self.index),
            "ready": self.index is not None,
        }
