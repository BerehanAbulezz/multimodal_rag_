"""
The single function the notebook actually calls: `ask(...)`.
Builds the index once (cached), optionally ingests a PDF, optionally
takes an image query, and runs everything through the corrective RAG loop.
"""
from .config import MAX_CORRECTIVE_RETRIES, TOP_K
from .corrective_rag import corrective_rag_answer
from .index_builder import build_index
from .pdf_ingest import ingest_pdf
from .retriever import Retriever

_retriever = None


def get_retriever() -> Retriever:
    """Builds the multimodal index on first call, then reuses it."""
    global _retriever
    if _retriever is None:
        items, matrix = build_index()
        _retriever = Retriever(items, matrix)
    return _retriever


def ask(
    query: str,
    image_path=None,
    pdf_path: str = None,
    top_k: int = TOP_K,
    max_retries: int = MAX_CORRECTIVE_RETRIES,
):
    """
    query:      the question, in text (required)
    image_path: optional - path or PIL.Image to use as the RETRIEVAL query
                (e.g. "what animal is this?" + a photo)
    pdf_path:   optional - a PDF to ingest into the index before answering
    """
    retriever = get_retriever()

    if pdf_path:
        n_chunks = ingest_pdf(pdf_path, retriever)
        print(f"[ingest] added {n_chunks} chunks from {pdf_path}")

    return corrective_rag_answer(
        query, retriever, top_k=top_k, max_retries=max_retries, query_image=image_path
    )
