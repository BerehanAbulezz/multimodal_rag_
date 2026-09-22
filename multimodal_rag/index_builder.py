"""
Builds the initial multimodal index:
  - one entry per row in data/animals.json          (type="text")
  - one entry per image in data/images/*.jpg          (type="image")
All entries are embedded into the same CLIP space so they can be mixed
in a single similarity search.
"""
import glob
import json
import os
import re

import numpy as np

from .config import DATA_JSON_PATH, IMAGES_DIR
from .embeddings import embed_image, embed_text


class MultimodalItem:
    """A single retrievable unit - either a text passage or an image."""

    def __init__(self, item_type: str, content, source: dict, embedding):
        self.type = item_type          # "text" or "image"
        self.content = content         # text string, or image file path
        self.source = source           # metadata dict (animal, topic, ...)
        self.embedding = embedding     # np.ndarray, L2-normalized

    def __repr__(self):
        tag = self.content[:40] if self.type == "text" else os.path.basename(self.content)
        return f"<MultimodalItem {self.type}: {tag!r}>"


def _animal_name_from_filename(path: str) -> str:
    base = os.path.splitext(os.path.basename(path))[0]   # e.g. "giraffe2"
    name = re.sub(r"\d+$", "", base)                       # -> "giraffe"
    return name.capitalize()


def build_index():
    """Returns (items: list[MultimodalItem], matrix: np.ndarray (N, D))."""
    items = []

    # --- text items from the JSON dataset --------------------------------
    with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    text_embs = embed_text(texts)
    for record, emb in zip(records, text_embs):
        items.append(MultimodalItem("text", record["text"], record, emb))

    # --- image items -------------------------------------------------------
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if image_paths:
        img_embs = embed_image(image_paths)
        for path, emb in zip(image_paths, img_embs):
            meta = {"animal": _animal_name_from_filename(path), "path": path}
            items.append(MultimodalItem("image", path, meta, emb))

    matrix = np.stack([it.embedding for it in items]).astype("float32")
    return items, matrix
