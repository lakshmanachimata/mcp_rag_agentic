"""Folder-based RAG index backed by ChromaDB and Ollama embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb

from ollama_client import embed_texts

RAG_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".log",
    ".xlsx",
    ".xls",
    ".xlsm",
}
TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".html",
    ".rst",
    ".xml",
}
SUPPORTED_EXTENSIONS = RAG_EXTENSIONS | TEXT_EXTENSIONS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / ".vectordb"


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _collection_name_for_folder(folder: Path) -> str:
    digest = hashlib.sha256(str(folder.resolve()).encode()).hexdigest()[:16]
    return f"rag_{digest}"


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def _read_csv(path: Path) -> str:
    import csv

    rows: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for row in csv.reader(handle):
            if row:
                rows.append("\t".join(row))
    return "\n".join(rows)


def _read_excel(path: Path) -> str:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
    parts: list[str] = []
    for sheet_name, frame in sheets.items():
        frame = frame.fillna("")
        lines = [
            "\t".join(str(value) for value in row if str(value).strip())
            for row in frame.values.tolist()
            if any(str(value).strip() for value in row)
        ]
        if lines:
            parts.append(f"Sheet: {sheet_name}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_file_content(path: Path) -> str | None:
    suffix = path.suffix.lower()
    readers = {
        ".pdf": _read_pdf,
        ".csv": _read_csv,
        ".xlsx": _read_excel,
        ".xls": _read_excel,
        ".xlsm": _read_excel,
        ".txt": _read_text_file,
        ".log": _read_text_file,
    }
    try:
        if suffix in readers:
            return readers[suffix](path)
        if suffix in TEXT_EXTENSIONS:
            return _read_text_file(path)
    except Exception:
        return None
    return None


def _load_documents(folder: Path) -> list[dict]:
    documents: list[dict] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = _read_file_content(path)
        if not content or not content.strip():
            continue

        relative = str(path.relative_to(folder))
        for index, chunk in enumerate(_chunk_text(content)):
            documents.append(
                {
                    "id": hashlib.sha1(f"{relative}:{index}".encode()).hexdigest(),
                    "text": chunk,
                    "metadata": {
                        "source": relative,
                        "chunk_index": index,
                    },
                }
            )
    return documents


class RagStore:
    def __init__(self, persist_dir: Path | str = DEFAULT_PERSIST_DIR) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    def build_from_folder(
        self,
        folder: Path | str,
        embedding_model: str,
        *,
        host: str | None = None,
    ) -> dict:
        folder_path = Path(folder).expanduser().resolve()
        if not folder_path.is_dir():
            raise ValueError(f"Folder does not exist: {folder_path}")

        documents = _load_documents(folder_path)
        if not documents:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(
                "No supported documents found in the selected folder. "
                f"Supported types: {supported}"
            )

        collection_name = _collection_name_for_folder(folder_path)
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

        collection = self._client.create_collection(name=collection_name)
        texts = [doc["text"] for doc in documents]
        embeddings = embed_texts(embedding_model, texts, host=host)

        collection.add(
            ids=[doc["id"] for doc in documents],
            documents=texts,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in documents],
        )

        sources = sorted({doc["metadata"]["source"] for doc in documents})
        return {
            "folder": str(folder_path),
            "collection_name": collection_name,
            "embedding_model": embedding_model,
            "file_count": len(sources),
            "chunk_count": len(documents),
            "sources": sources,
            "persist_dir": str(self.persist_dir),
        }

    def get_details(self, collection_name: str) -> dict | None:
        try:
            collection = self._client.get_collection(collection_name)
        except Exception:
            return None

        count = collection.count()
        sample = collection.get(limit=1, include=["metadatas"])
        folder = ""
        if sample.get("metadatas"):
            folder = sample["metadatas"][0].get("source", "")

        all_meta = collection.get(include=["metadatas"])
        sources = sorted(
            {
                meta.get("source", "")
                for meta in (all_meta.get("metadatas") or [])
                if meta.get("source")
            }
        )

        return {
            "collection_name": collection_name,
            "chunk_count": count,
            "file_count": len(sources),
            "sources": sources,
            "persist_dir": str(self.persist_dir),
            "sample_source": folder,
        }

    def query(
        self,
        collection_name: str,
        query_text: str,
        embedding_model: str,
        *,
        top_k: int = 5,
        host: str | None = None,
    ) -> list[dict]:
        collection = self._client.get_collection(collection_name)
        query_embedding = embed_texts(embedding_model, [query_text], host=host)[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        matches: list[dict] = []
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "text": document,
                    "source": metadata.get("source", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "distance": distance,
                }
            )
        return matches
