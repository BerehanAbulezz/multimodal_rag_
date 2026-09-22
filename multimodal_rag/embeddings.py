"""
Joint text/image embedding space using CLIP.

Both `embed_text` and `embed_image` return L2-normalized vectors living in
the SAME space, which is what makes cross-modal retrieval (text query ->
image results, image query -> text results) possible.
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


def embed_text(texts):
    """texts: list[str] -> np.ndarray of shape (N, D), L2-normalized."""
    model, processor = _load()
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
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
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()
