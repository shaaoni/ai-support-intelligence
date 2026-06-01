# ai-support-intelligence

# AI Support Intelligence Platform

A client-deployable RAG agent that ingests support documents and ticket history,
answers customer queries, routes issues via an ML classifier, and exposes a live
observability dashboard.

Built as a capstone project for the **Forward Deployed AI Engineering** course.

---

## Architecture

```
User query → Streamlit UI
               → LangGraph agent
                   → classify intent   (sklearn FastAPI)
                   → retrieve & answer (LangChain + ChromaDB)
                   → escalate          (human handoff)
               → Guardrails check
               → LangSmith trace
```

## Setup

```bash
# 1. Clone and enter
git clone <your-repo-url>
cd ai-support-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install base packages (add more each phase)
pip install python-dotenv pandas

# 4. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 5. Verify setup
python scripts/health_check.py
```

## Run the app (Phase 6)

```bash
streamlit run app/main.py
```

## Project structure

```
src/
  pipeline/   ETL ingestion + Great Expectations quality checks
  rag/        Chunking, embeddings, ChromaDB vector store
  ml/         Sklearn classifier + MLflow tracking + FastAPI server
  agent/      LangGraph orchestration + guardrails
  prompts/    Versioned prompt templates + loader
app/          Streamlit UI (chat, dashboard, admin)
data/
  raw/        Source CSVs and Markdown docs (not in git)
  chroma/     Persisted vector DB (not in git)
  expectations/ Great Expectations suites
scripts/      Utility scripts (health check, pipeline runner)
docs/         Discovery brief, architecture notes
tests/        Unit tests per module
```

## Modules covered

| Phase | Module in course | What's built |
|-------|-----------------|--------------|
| 1 | M2 — Data & Pipeline | Ingestion + quality layer |
| 2 | M4 — ML Lifecycle | Classifier + MLflow + FastAPI |
| 3 | M5 — NLP & LLMs | Embeddings + ChromaDB |
| 4 | M3 — Prompt Engineering | Versioned prompt templates |
| 5 | M6 — Agents | LangGraph + guardrails |
| 6 | M6 — Deployment | Streamlit UI + observability |

## Data source

[Kaggle Customer Support Twitter Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
