"""
Central configuration for the multimodal RAG package.

Nothing here talks to the network on import - the Gemini client is created
lazily via `init_client(api_key)`, which the notebook calls once it has
pulled the API key from Colab secrets.
"""
import os

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# --- Models -----------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
GEMINI_MODEL = "gemini-3.6-flash"

# --- Paths --------------------------------------------------------------
# repo_root/multimodal_rag/config.py -> repo_root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON_PATH = os.path.join(REPO_ROOT, "data", "animals.json")
IMAGES_DIR = os.path.join(REPO_ROOT, "data", "images")

# --- Retrieval / chunking ------------------------------------------------
TOP_K = 3
MAX_CORRECTIVE_RETRIES = 2

# CLIP's text tower truncates at 77 tokens (~ a short sentence or two),
# so PDF chunks must stay small - unlike the old sentence-transformers
# pipeline, which tolerated much longer chunks.
PDF_CHUNK_SIZE = 250
PDF_CHUNK_OVERLAP = 30

# --- Gemini client (set once from the notebook) --------------------------
client = None


def init_client(api_key: str):
    """Create the Gemini client. Call this once from the notebook after
    reading the API key (e.g. from google.colab.userdata)."""
    global client
    from google import genai
    client = genai.Client(api_key=api_key)
    return client
