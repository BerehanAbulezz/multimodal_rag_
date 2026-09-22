"""
Joint text/image embedding space using CLIP.

Both `embed_text` and `embed_image` return L2-normalized vectors living in
the SAME space, which is what makes cross-modal retrieval (text query ->
image results, image query -> text results) possible.

Implementation note: `model.get_text_features(...)` / `get_image_features(...)`
have an unstable return shape across `transformers` versions (plain tensor
vs. various ModelOutput variants), which is what caused earlier crashes.
To avoid depending on that, we call the internal `model.text_model` /
`model.vision_model` encoders directly and project their `pooler_output`
ourselves with `model.text_projection` / `model.visual_projection`. That
internal API has been stable for years and isn't affected by changes to
the convenience wrappers.
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
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        pooled_output = text_outputs.pooler_output
        feats = model.text_projection(pooled_output)
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
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"])
        pooled_output = vision_outputs.pooler_output
        feats = model.visual_projection(pooled_output)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()
