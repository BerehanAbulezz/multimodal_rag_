# Multimodal RAG (CLIP + BLIP + Corrective RAG)

## هيكل المشروع
```
multimodal_rag_repo/
├── multimodal_rag/
│   ├── __init__.py        # exposes ask(), get_retriever(), init_client()
│   ├── config.py           # models, paths, TOP_K, lazy Gemini client
│   ├── embeddings.py        # CLIP: embed_text(), embed_image()
│   ├── captioning.py        # BLIP: generate_caption()
│   ├── index_builder.py     # builds the initial text+image index
│   ├── retriever.py         # Retriever.retrieve() - unified text/image search
│   ├── pdf_ingest.py         # extract_text, chunk_text, ingest_pdf()
│   ├── generator.py          # build_prompt(), generate_answer() (Gemini)
│   └── corrective_rag.py     # grade_answer(), corrective_rag_answer()
├── data/
│   ├── animals.json          # 50-entry text dataset (dog/cat/giraffe/elephant)
│   └── images/                # 12 animal photos
├── requirements.txt
└── ragmini_multimodal.ipynb   # the Colab notebook (thin - just calls the package)
```

## إزاي ترفعه على GitHub (مرة واحدة بس)

1. اعمل repo فاضي جديد على github.com (من غير README ولا .gitignore).
2. من جوه المجلد ده على جهازك:
   ```bash
   cd multimodal_rag_repo
   git init
   git add .
   git commit -m "initial multimodal RAG package"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo-name>.git
   git push -u origin main
   ```
3. افتح `ragmini_multimodal.ipynb` وغيّر متغيّر `REPO_URL` في أول خلية بالرابط بتاع الـ repo بتاعك.

## بعد كده
أي تعديل تعمله تاني، تكفي:
```bash
git add . && git commit -m "..." && git push
```
وفي الـ Colab تعمل `!git pull` بدل ما تعمل clone من الأول.
