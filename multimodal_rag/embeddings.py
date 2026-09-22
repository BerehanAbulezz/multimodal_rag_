"""
Joint text/image embedding space using CLIP.

Both `embed_text` and `embed_image` return L2-normalized vectors living in
the SAME space, which is what makes cross-modal retrieval (text query ->
image results, image query -> text results) possible.

Note: depending on the installed `transformers` version,
`model.get_text_features(...)` / `get_image_features(...)` can return:
  (a) a plain tensor, already projected - older versions, or
  (b) a ModelOutput with an already-projected `text_embeds`/`image_embeds`
      field - some newer versions, or
  (c) a ModelOutput with only a raw `pooler_output` that still needs to be
      passed through `model.text_projection` / `model.visual_projection`.
`_extract_features` below tries (a), then (b), then falls back to (c),
instead of assuming a single fixed shape.
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


def _extract_features(output, embeds_attr: str, projection):
    """Return a plain (batch, dim) tensor of ALREADY-PROJECTED features."""
    # (a) already a tensor - assume it's already projected
    if torch.is_tensor(output):
        return output

    # (b) ModelOutput already carries the projected embedding
    embeds = getattr(output, embeds_attr, None)
    if embeds is not None:
        return embeds

    # (c) fall back: pool + project manually
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        pooled = output[1]  # ModelOutput behaves like a tuple
    return projection(pooled)


def embed_text(texts):
    """texts: list[str] -> np.ndarray of shape (N, D), L2-normalized."""
    model, processor = _load()
    inputs = processor(
        text=texts, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        raw = model.get_text_features(**inputs)
        feats = _extract_features(raw, "text_embeds", model.text_projection)
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
        feats = _extract_features(raw, "image_embeds", model.visual_projection)
    feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
    return feats.cpu().numpy()
