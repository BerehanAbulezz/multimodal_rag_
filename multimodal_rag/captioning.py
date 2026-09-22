"""
BLIP image captioning.

Used in two places:
1. generator.py - turns a retrieved image into a text description so the
   LLM has something to read when building its answer.
2. corrective_rag.py (indirectly, via the grader) - the caption becomes
   part of the context the grader checks the answer against.
"""
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from .config import BLIP_MODEL_NAME, DEVICE

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _processor = BlipProcessor.from_pretrained(BLIP_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_NAME).to(DEVICE)
        _model.eval()
    return _model, _processor


def generate_caption(image, max_new_tokens: int = 40) -> str:
    """image: file path or PIL.Image -> short text caption."""
    model, processor = _load()
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(out[0], skip_special_tokens=True)
