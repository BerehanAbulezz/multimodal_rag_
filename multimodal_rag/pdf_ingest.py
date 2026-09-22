"""
Same extract/chunk logic as the original notebook, but chunks get embedded
with the CLIP text encoder (not sentence-transformers) so they land in the
same space as the animal dataset and can be added to a live Retriever.
"""
from pypdf import PdfReader

from .config import PDF_CHUNK_OVERLAP, PDF_CHUNK_SIZE
from .embeddings import embed_text


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def chunk_text(text: str, chunk_size: int = PDF_CHUNK_SIZE, overlap: int = PDF_CHUNK_OVERLAP):
    text = " ".join(text.split())  # normalize whitespace
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_path: str, retriever) -> int:
    """Extract + chunk + embed a PDF and append it to `retriever`'s index.
    Returns the number of chunks added."""
    from .index_builder import MultimodalItem  # local import avoids a cycle

    raw_text = extract_text(pdf_path)
    chunks = chunk_text(raw_text)
    if not chunks:
        return 0

    embs = embed_text(chunks)
    items = [
        MultimodalItem("text", chunk, {"source_file": pdf_path, "animal": None}, emb)
        for chunk, emb in zip(chunks, embs)
    ]
    retriever.add_items(items, embs)
    return len(chunks)
