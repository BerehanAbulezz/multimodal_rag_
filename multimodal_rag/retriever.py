"""
A thin wrapper around the (items, matrix) pair produced by index_builder.

`retrieve` accepts either a text query or an image query and always
searches the SAME matrix, so results can be a mix of text passages and
images ranked by cosine similarity.
"""
import numpy as np

from .embeddings import embed_image, embed_text


class Retriever:
    def __init__(self, items, matrix: np.ndarray):
        self.items = items
        self.matrix = matrix   # shape (N, D), rows L2-normalized

    def add_items(self, new_items, new_embs: np.ndarray):
        """Used by pdf_ingest.py to extend the index at runtime."""
        self.items.extend(new_items)
        self.matrix = np.vstack([self.matrix, new_embs.astype("float32")])

    def retrieve(self, query, query_type: str = "text", top_k: int = 3, item_filter=None):
        """
        query: str (if query_type="text") or image path/PIL.Image (if "image")
        item_filter: optional function(MultimodalItem) -> bool, e.g. to
                     restrict results to only images or only text.
        Returns: list[(MultimodalItem, score)], best first.
        """
        if query_type == "text":
            q_emb = embed_text([query])[0]
        elif query_type == "image":
            q_emb = embed_image([query])[0]
        else:
            raise ValueError("query_type must be 'text' or 'image'")

        scores = self.matrix @ q_emb
        ranked_idx = np.argsort(scores)[::-1]

        if item_filter is not None:
            ranked_idx = [i for i in ranked_idx if item_filter(self.items[i])]

        top_idx = ranked_idx[:top_k]
        return [(self.items[i], float(scores[i])) for i in top_idx]
