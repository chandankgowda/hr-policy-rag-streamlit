from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
DENSE_WEIGHT = 0.65


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    page: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    dense_score: float
    bm25_score: float
    hybrid_score: float


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple paragraph-aware character chunker with overlap."""
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}".strip()
        else:
            start = 0
            while start < len(para):
                end = min(len(para), start + chunk_size)
                chunks.append(para[start:end])
                if end == len(para):
                    current = ""
                    break
                start = max(end - overlap, start + 1)

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def extract_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        pages.append((page_no, clean_text(page.extract_text() or "")))
    return pages


def build_chunks(pdf_bytes: bytes) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 0
    for page_no, text in extract_pdf(pdf_bytes):
        for piece in split_text(text):
            chunks.append(Chunk(chunk_id=chunk_id, page=page_no, text=piece))
            chunk_id += 1
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def minmax(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if np.isclose(lo, hi):
        return np.ones_like(values, dtype=np.float32) if hi > 0 else np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


class HybridRAGIndex:
    def __init__(self, chunks: list[Chunk], embedder: SentenceTransformer):
        if not chunks:
            raise ValueError("No readable text was found in the PDF.")

        self.chunks = chunks
        self.embedder = embedder
        self.texts = [c.text for c in chunks]

        embeddings = embedder.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        self.embeddings = embeddings
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def search(self, query: str, top_k: int = TOP_K, dense_weight: float = DENSE_WEIGHT) -> list[SearchHit]:
        q_emb = self.embedder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        dense_scores_all = (self.embeddings @ q_emb[0]).astype(np.float32)
        bm25_scores_all = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

        dense_norm = minmax(dense_scores_all)
        bm25_norm = minmax(bm25_scores_all)
        hybrid = dense_weight * dense_norm + (1.0 - dense_weight) * bm25_norm

        k = min(top_k, len(self.chunks))
        top_ids = np.argsort(-hybrid)[:k]

        return [
            SearchHit(
                chunk=self.chunks[int(i)],
                dense_score=float(dense_norm[int(i)]),
                bm25_score=float(bm25_norm[int(i)]),
                hybrid_score=float(hybrid[int(i)]),
            )
            for i in top_ids
        ]


def format_context(hits: Iterable[SearchHit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(
            f"[Page {hit.chunk.page}]\n{hit.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def pdf_fingerprint(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]
