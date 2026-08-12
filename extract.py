import os
import re
from typing import Any

import fitz

_HYPHEN_WRAP = re.compile(r"(\w)-\n(\w)")
_MANY_BREAKS = re.compile(r"\n{3,}")
_SOFT_BREAK = re.compile(r"(?<!\n)\n(?!\n)")
_INLINE_SPACE = re.compile(r"[^\S\n]+")
_PADDED_BREAK = re.compile(r" *\n *")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _HYPHEN_WRAP.sub(r"\1\2", text)
    text = _MANY_BREAKS.sub("\n\n", text)
    text = _SOFT_BREAK.sub(" ", text)
    text = _INLINE_SPACE.sub(" ", text)
    text = _PADDED_BREAK.sub("\n", text)
    return text.strip()


def extract_pages(pdf_path: str) -> list[dict[str, Any]]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    pages: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as doc:
        total = doc.page_count
        title = (doc.metadata or {}).get("title") or filename
        for number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if not text:
                continue
            pages.append({
                "filename": filename,
                "title": title,
                "page": number,
                "total_pages": total,
                "words": len(text.split()),
                "text": text,
            })

    return pages


def extract_many(pdf_paths: list[str]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    for path in pdf_paths:
        try:
            pages.extend(extract_pages(path))
        except Exception as error:
            print(f"Skipped {os.path.basename(path)}: {error}")
    return pages


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    by_file: dict[str, dict[str, Any]] = {}
    for page in pages:
        entry = by_file.setdefault(page["filename"], {
            "filename": page["filename"],
            "title": page["title"],
            "total_pages": page["total_pages"],
            "indexed_pages": 0,
            "words": 0,
        })
        entry["indexed_pages"] += 1
        entry["words"] += page["words"]
    return {"documents": list(by_file.values()), "pages": len(pages)}


def find_pdfs(target: str) -> list[str]:
    found: list[str] = []
    for part in (p.strip().strip('"') for p in target.split(",")):
        if not part:
            continue
        if os.path.isfile(part) and part.lower().endswith(".pdf"):
            found.append(os.path.abspath(part))
        elif os.path.isdir(part):
            for root, _, files in os.walk(part):
                found += [
                    os.path.abspath(os.path.join(root, f))
                    for f in files
                    if f.lower().endswith(".pdf")
                ]
    return sorted(set(found))
