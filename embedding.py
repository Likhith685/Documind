import os
from typing import Any, Optional

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

PRIMARY_MODEL = "BAAI/bge-small-en"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

_cached_model: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(name: str = PRIMARY_MODEL, device: str = "cpu") -> HuggingFaceEmbeddings:
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    options = {
        "model_kwargs": {"device": device},
        "encode_kwargs": {"normalize_embeddings": True},
    }
    try:
        _cached_model = HuggingFaceEmbeddings(model_name=name, **options)
    except Exception as error:
        print(f"Falling back to {FALLBACK_MODEL}: {error}")
        _cached_model = HuggingFaceEmbeddings(model_name=FALLBACK_MODEL, **options)
    return _cached_model


def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> tuple[list[str], list[dict[str, Any]]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for page in pages:
        text = page.get("text", "")
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        for index, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "source": page.get("filename", "unknown.pdf"),
                "title": page.get("title", page.get("filename", "unknown.pdf")),
                "page": page.get("page", 1),
                "total_pages": page.get("total_pages", 1),
                "chunk": index,
                "chunks_on_page": len(chunks),
            })

    return texts, metadatas


def build_index(pages: list[dict[str, Any]], path: str) -> FAISS:
    texts, metadatas = chunk_pages(pages)
    if not texts:
        raise ValueError("No readable text found in these PDFs.")

    index = FAISS.from_texts(texts, get_embedding_model(), metadatas=metadatas)
    save_index(index, path)
    return index


def extend_index(index: Optional[FAISS], pages: list[dict[str, Any]], path: str) -> FAISS:
    if index is None:
        return build_index(pages, path)

    texts, metadatas = chunk_pages(pages)
    if not texts:
        raise ValueError("No readable text found in these PDFs.")

    index.add_texts(texts, metadatas=metadatas)
    save_index(index, path)
    return index


def save_index(index: FAISS, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    index.save_local(path)


def load_index(path: str) -> Optional[FAISS]:
    if not os.path.exists(os.path.join(path, "index.faiss")):
        return None
    return FAISS.load_local(path, get_embedding_model(), allow_dangerous_deserialization=True)


def index_size(index: Optional[FAISS]) -> int:
    if index is None:
        return 0
    return int(index.index.ntotal)
