# Multimodal RAG (CLIP + BLIP + Corrective RAG)

A retrieval-augmented QA pipeline that can be queried with **text, an image, or a PDF**,
retrieves from a single shared embedding space, and self-corrects when its own answer
doesn't hold up.

## How it works

**1. Shared embedding space (CLIP).**
Text passages and images are both embedded with `openai/clip-vit-base-patch32` into
the same vector space, so a single similarity search can return a mix of text and
images for either a text query or an image query.

**2. Initial index.**
Built from `data/animals.json` (50 short text passages about dogs, cats, giraffes,
and elephants) and `data/images/` (12 photos of those animals).

**3. Image captioning (BLIP).**
`Salesforce/blip-image-captioning-base` turns any retrieved (or query) image into a
short text caption, so a text-only LLM can read what an image shows. For image
queries specifically, the pipeline runs a *hybrid* search: the query image is
captioned and that caption drives a text-mode search (for relevant facts), combined
with a direct image-mode search (for the closest matching photo) — pure
image-embedding search alone tends to surface visually similar photos rather than
the topical text an answer actually needs.

**4. PDF ingestion.**
Any PDF can be uploaded at query time. It's text-extracted (`pypdf`), chunked, embedded
with the same CLIP text encoder, and appended to the live index — so its content
becomes searchable right alongside the animal dataset for the rest of the session.

**5. Generation (Gemini).**
Retrieved text passages and image captions are assembled into a context block and
passed to Gemini to produce the final answer.

**6. Corrective RAG loop.**
After generating an answer, a grader call asks Gemini whether the answer is (a) fully
supported by the retrieved context and (b) actually complete — addresses every part
of the question. If not, the query is rewritten (text queries) or retrieval is widened
(image queries) and the cycle repeats, up to a configurable number of retries. The
last attempt is always returned, flagged with a `warning` if it never passed grading.

## Repository structure

```
multimodal_rag_repo/
├── multimodal_rag/
│   ├── __init__.py         # exposes ask(), get_retriever(), init_client()
│   ├── config.py            # model names, paths, TOP_K, lazy Gemini client
│   ├── embeddings.py         # CLIP: embed_text(), embed_image()
│   ├── captioning.py         # BLIP: generate_caption()
│   ├── index_builder.py      # builds the initial text+image index
│   ├── retriever.py          # Retriever.retrieve() - unified text/image search
│   ├── pdf_ingest.py          # extract_text, chunk_text, ingest_pdf()
│   ├── generator.py           # build_prompt(), generate_answer() (Gemini)
│   └── corrective_rag.py      # grade_answer(), corrective_rag_answer()
├── data/
│   ├── animals.json           # 50-entry text dataset (dog/cat/giraffe/elephant)
│   └── images/                 # 12 animal photos
├── requirements.txt
└── ragmini_multimodal.ipynb    # thin Colab notebook - just calls the package
```

## Using this repo

### 1. Get an API key
You need a Google Gemini API key. Get one from [Google AI Studio](https://aistudio.google.com/).

### 2. Open the notebook in Colab
Open `ragmini_multimodal.ipynb` in Google Colab (or upload it there).

### 3. Add your API key to Colab Secrets
In Colab, click the key icon  in the left sidebar → add a secret named
`GOOGLE_API_KEY` with your key as the value, and enable notebook access for it.

### 4. Point the notebook at this repo
In the notebook's second code cell, set:
```python
REPO_URL = "https://github.com/<your-username>/<your-repo>.git"
REPO_DIR = "<your-repo-folder-name>"
```

### 5. Run the cells top to bottom
The notebook will clone the repo, install dependencies, initialize the Gemini
client, then walk through examples: a text query, an image query, and PDF
ingestion (summarizing a PDF, asking a specific question about it, and comparing
its content with the existing animal dataset).

### 6. Use it in your own code
Once the package is on `sys.path`, the whole pipeline is one function call:
```python
import multimodal_rag as mrag
mrag.init_client(GOOGLE_API_KEY)

# Text query
result = mrag.ask("What do giraffes eat?")

# Image query
result = mrag.ask("What animal is this?", image_path="path/to/photo.jpg")

# PDF query (ingests the PDF into the index, then answers)
result = mrag.ask("Summarize this document.", pdf_path="path/to/file.pdf")

# result = {"answer": str, "results": [...], "attempts": int, "trace": [...], "warning"?: str}
```

### Updating the package
If you edit any file under `multimodal_rag/`, commit and push as usual:
```bash
git add .
git commit -m "..."
git push
```
Then in Colab, re-run the clone/pull cell (`!git pull` if already cloned) and
**Runtime → Restart session** before re-running the rest of the notebook, since
models and modules are cached in memory for the duration of the session.
