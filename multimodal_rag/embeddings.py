"""
Joint text/image embedding space using CLIP.

Both `embed_text` and `embed_image` return L2-normalized vectors living in
the SAME space, which is what makes cross-modal retrieval (text query ->
image results, image query -> text results) possible.

Note: depending on the installed `transformers` version,
`model.get_text_features(...)` / `get_image_features(...)` either return a
plain tensor (older versions) or a ModelOutput object such as
BaseModelOutputWithPooling (newer versions). `_extract_features` below
handles both cases instead of assuming one.
"""
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import CLIP_MODEL_NAME, DEVICE

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _model.eval()
    return _model, _processor


def _extract_features(output, projection):
    """Return a plain (batch, dim) tensor whether `output` already is one,
    or is a ModelOutput we need to pool + project ourselves."""
    if torch.is_tensor(output):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        # ModelOutput behaves like a tuple: (last_hidden_state, pooler_output, ...)
        pooled = output[1]
    return projection(pooled)


def embed_text(texts):
    """texts: list[str] -> np.ndarray of shape (N, D), L2-normalized."""
    model, processor = _load()
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        raw = model.get_text_features(**inputs)
        feats = _extract_features(raw, model.text_projection)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()


def embed_image(images):
    """images: list[str paths] or list[PIL.Image] -> np.ndarray (N, D)."""
    model, processor = _load()
    pil_images = []
    for img in images:
        if isinstance(img, str):
            pil_images.append(Image.open(img).convert("RGB"))
        else:
            pil_images.append(img.convert("RGB"))
    inputs = processor(images=pil_images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        raw = model.get_image_features(**inputs)
        feats = _extract_features(raw, model.visual_projection)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()