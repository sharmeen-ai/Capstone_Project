# 🧠 AI Knowledge Assistant

A document Q&A web app: upload `.txt` or `.pdf` files, then ask questions and
get answers grounded in that content, with sources cited. Built with **Flask**,
**LangChain**, **FAISS**, and **Google Gemini**.

*(Add your own screenshots — see the [Screenshots](#screenshots) section below.)*

## Features

- 📤 Drag-and-drop upload for `.txt` and `.pdf` documents
- 🔍 Retrieval-Augmented Generation (RAG): answers are generated only from
  the content of your uploaded documents, with source filenames cited
- 💬 Simple chat-style interface, no page reloads
- 🗑️ Delete individual documents or clear everything and start fresh
- 📊 Optional script to plot training/evaluation metrics if you're using this
  as part of a model-evaluation workflow (`plot_model_metrics.py`)

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask, Flask-CORS |
| LLM | Google Gemini (`gemini-2.5-flash`) via `langchain-google-genai` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs locally) |
| Vector store | FAISS |
| Orchestration | LangChain (`RetrievalQA`) |
| Frontend | Single static HTML/CSS/JS page, no build step |

## How it works

1. You upload a document. Flask saves it to `uploads/`.
2. The document is split into ~1000-character chunks and embedded locally
   using a small HuggingFace sentence-transformer model.
3. Chunks are indexed in a FAISS vector store.
4. When you ask a question, the most relevant chunks are retrieved and sent
   to Gemini along with your question, and the answer plus source filenames
   are returned.

## Screenshots

> **For contributors/students reusing this repo:** run the app locally
> (steps below), open `http://127.0.0.1:5001`, upload a document, ask it a
> question, then take a screenshot and save it into the `screenshots/`
> folder using the filenames below — the images will then show up
> automatically here and in this README.

| File | Shows |
|---|---|
| `screenshots/main-ui.png` | The main screen before any documents are uploaded |
| `screenshots/upload.png` | A document uploaded and listed in the sidebar |
| `screenshots/chat.png` | A question asked with an answer and cited sources shown |

```
screenshots/
├── main-ui.png
├── upload.png
└── chat.png
```

## Project structure

```
.
├── app.py                   # Flask backend + RAG pipeline
├── index.html                # Frontend (served by Flask at "/")
├── plot_model_metrics.py    # Optional: plot training/eval metrics from a CSV
├── requirements.txt
├── .env.example             # Template for your API key — copy to .env
├── .gitignore
├── LICENSE
├── uploads/                  # Uploaded documents land here (gitignored)
├── plots/                    # Output of plot_model_metrics.py (gitignored)
└── screenshots/              # README images (see above)
```

## Setup

Requires Python 3.10+ (developed/tested on 3.12 and 3.14).

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Get a free Gemini API key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in → **Get API Key** → **Create API key**
3. Copy `.env.example` to `.env` and paste your key in:
   ```
   GOOGLE_API_KEY=your_key_here
   ```

> ⚠️ Never commit your real `.env` file. Only `.env.example` (with no real
> key) should ever be pushed to GitHub. If a real key is ever exposed, treat
> it as compromised and generate a new one immediately.

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5001** in your browser — Flask serves the frontend
directly, so there's nothing else to start.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend |
| `POST` | `/upload` | Upload a `.txt`/`.pdf` file (multipart form, field name `file`) |
| `GET` | `/documents` | List uploaded documents |
| `DELETE` | `/delete/<filename>` | Delete one document and reindex the rest |
| `POST` | `/clear` | Delete all documents and reset the index |
| `POST` | `/query` | `{"question": "..."}` → `{"answer": "...", "sources": [...]}` |
| `GET` | `/health` | Health check |

## Limitations & notes

- Max upload size is 16MB.
- Every upload/delete rebuilds the FAISS index from *all* remaining
  documents — fine for small document sets, but it re-embeds everything
  each time rather than incrementally.
- The embedding model downloads once from Hugging Face on first use
  (needs internet the first time only).
- This runs on Flask's development server, which is fine for local/demo use.
  For anything public-facing, put it behind a real WSGI server (e.g.
  `gunicorn app:app`) and restrict CORS to your actual frontend origin
  instead of `*`.

## Model evaluation plots (optional)

If you're using this project alongside a model-training workflow,
`plot_model_metrics.py` generates scatter plots from a metrics CSV
(`epoch`, `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`, and
optionally `predictions`/`targets`):

```bash
python plot_model_metrics.py --metrics metrics.csv --output-dir plots
# or, for a quick demo with synthetic data:
python plot_model_metrics.py --example
```

## License

MIT — see [LICENSE](LICENSE). Feel free to reuse or adapt for your own
capstone/project.
