from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

from rag_engine import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DENSE_WEIGHT,
    EMBEDDING_MODEL,
    TOP_K,
    HybridRAGIndex,
    build_chunks,
    format_context,
    pdf_fingerprint,
)


APP_TITLE = "HR Policy Q&A | Hybrid RAG Demo"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEMO_PDF = Path("data/demo_hr_policy.pdf")

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.8rem;}
.small-muted {color: #6b7280; font-size: 0.9rem;}
.result-card {padding: 1.0rem 1.1rem; border: 1px solid #e5e7eb; border-radius: 12px;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading MiniLM embedding model...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Building hybrid FAISS + BM25 index...")
def build_index(pdf_bytes: bytes, _fingerprint: str) -> HybridRAGIndex:
    chunks = build_chunks(pdf_bytes)
    return HybridRAGIndex(chunks, get_embedder())


def get_api_key() -> str:
    try:
        return str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        return os.getenv("GROQ_API_KEY", "").strip()


def get_model_name() -> str:
    try:
        return str(st.secrets.get("GROQ_MODEL", DEFAULT_MODEL)).strip()
    except Exception:
        return os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip()


def generate_answer(question: str, context: str, api_key: str, model: str) -> str:
    client = Groq(api_key=api_key)
    system = (
        "You are an enterprise HR policy assistant. Answer only from the supplied policy evidence. "
        "Do not invent company policy. Cite every policy claim using [Page N]. "
        "If the evidence is incomplete or conflicting, say what cannot be confirmed and advise HR escalation. "
        "Keep the answer concise, practical, and employee-friendly."
    )
    user = f"POLICY EVIDENCE:\n{context}\n\nEMPLOYEE QUESTION:\n{question}"

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=500,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


st.title("✈️ Enterprise HR Policy Q&A Assistant — Hybrid RAG Demo")
st.caption("Hybrid RAG demo: MiniLM embeddings + FAISS dense retrieval + BM25 lexical retrieval + grounded LLM generation")

with st.sidebar:
    st.header("Demo controls")
    source = st.radio(
        "Policy source",
        ["Public-safe demo policy", "Upload a PDF for this session"],
        help="The repository ships only with a synthetic demo policy. Your original project handbook should stay private unless you have permission to publish it.",
    )

    if source == "Upload a PDF for this session":
        uploaded = st.file_uploader("Upload HR policy PDF", type=["pdf"])
        pdf_bytes = uploaded.getvalue() if uploaded else None
    else:
        pdf_bytes = DEMO_PDF.read_bytes() if DEMO_PDF.exists() else None

    st.divider()
    st.subheader("RAG configuration")
    st.code(
        f"Chunk size: {CHUNK_SIZE}\n"
        f"Overlap: {CHUNK_OVERLAP}\n"
        f"Top-K: {TOP_K}\n"
        f"Dense/BM25 weight: {DENSE_WEIGHT:.2f}/{1-DENSE_WEIGHT:.2f}\n"
        "Reranking: Off",
        language="text",
    )

    api_key = get_api_key()
    model = get_model_name()
    if api_key:
        st.success("Groq API key detected")
    else:
        st.warning("Add GROQ_API_KEY in Streamlit Secrets to enable answer generation.")
    st.caption(f"Model: `{model}`")

    st.divider()
    st.markdown("**Public demo note**")
    st.caption(
        "The bundled policy is synthetic and is not an official airline or course handbook. "
        "Use private/company PDFs only when you have permission."
    )

if not pdf_bytes:
    st.info("Choose the demo policy or upload a PDF to begin.")
    st.stop()

try:
    fp = pdf_fingerprint(pdf_bytes)
    rag_index = build_index(pdf_bytes, fp)
except Exception as exc:
    st.error(f"Could not build the index: {exc}")
    st.stop()

metric_cols = st.columns(4)
metric_cols[0].metric("Chunks", len(rag_index.chunks))
metric_cols[1].metric("Retriever", "Hybrid")
metric_cols[2].metric("Top-K", TOP_K)
metric_cols[3].metric("Dense weight", f"{DENSE_WEIGHT:.2f}")

sample_questions = [
    "What happens if my probation is extended?",
    "I have a family bereavement. How should I notify the office and what leave may apply?",
    "How should I report suspected workplace harassment?",
]

st.subheader("Ask a policy question")
selected = st.selectbox("Try a sample question", ["Custom question"] + sample_questions)
default_q = "" if selected == "Custom question" else selected
question = st.text_area("Question", value=default_q, height=90, placeholder="Type an HR policy question...")

ask = st.button("Ask the policy assistant", type="primary", use_container_width=False)

if ask:
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    hits = rag_index.search(question.strip())
    context = format_context(hits)

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown("### Grounded answer")
        if not api_key:
            st.info(
                "Retrieval is working, but answer generation is disabled until `GROQ_API_KEY` is added in Streamlit Secrets. "
                "The retrieved policy evidence is shown on the right."
            )
        else:
            try:
                with st.spinner("Generating a grounded answer..."):
                    answer = generate_answer(question.strip(), context, api_key, model)
                st.markdown(answer)
            except Exception as exc:
                st.error(f"LLM request failed: {exc}")

        st.caption("This demo provides policy lookup assistance, not a final HR decision.")

    with right:
        st.markdown("### Retrieved evidence")
        rows = []
        for rank, hit in enumerate(hits, start=1):
            rows.append(
                {
                    "Rank": rank,
                    "Page": hit.chunk.page,
                    "Hybrid": round(hit.hybrid_score, 3),
                    "Dense": round(hit.dense_score, 3),
                    "BM25": round(hit.bm25_score, 3),
                }
            )
            with st.expander(f"#{rank} · Page {hit.chunk.page} · score {hit.hybrid_score:.3f}"):
                st.write(hit.chunk.text)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

st.divider()
with st.expander("Architecture & project positioning"):
    st.markdown(
        """
**Flow:** PDF → page extraction → paragraph-aware chunking → MiniLM embeddings → FAISS + BM25 → weighted hybrid retrieval → Top-K evidence → Groq-hosted LLM → grounded answer with page citations.

This deployment intentionally uses the best balanced configuration from the project experiment: **1000-character chunks, 150-character overlap, Top-K 4, dense weight 0.65, no reranking, grounded prompt**.
"""
    )
