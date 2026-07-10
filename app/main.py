import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.agent.graph import SupportGraph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Support Intelligence",
    page_icon="🤖",
    layout="wide",
)

# ── Shared graph instance ─────────────────────────────────────────────────────
@st.cache_resource
def get_graph():
    return SupportGraph()

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigation",
    ["💬 Chat", "📊 Pipeline Health", "🛠️ Admin"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Support Intelligence**")
st.sidebar.markdown("Phase 6 — FDE Demo")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.title("💬 AI Support Chat")
    st.markdown("Ask any support question. The agent will classify, retrieve, and answer.")

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "run_logs" not in st.session_state:
        st.session_state.run_logs = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Route",      meta.get("route", "—"))
                col2.metric("Category",   meta.get("category", "—"))
                col3.metric("Confidence", f"{meta.get('confidence', 0):.0%}")
                col4.metric("Latency",    f"{meta.get('total_ms', 0):.0f}ms")

    # Input
    if prompt := st.chat_input("Type your support question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                graph = get_graph()
                start = time.time()
                state = graph.invoke(prompt, mock=True)
                total_ms = round((time.time() - start) * 1000)

            answer = state["final_answer"]
            st.markdown(answer)

            meta = {
                "route":      state["route"],
                "category":   state["category"],
                "confidence": state["confidence"],
                "total_ms":   total_ms,
            }

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Route",      meta["route"])
            col2.metric("Category",   meta["category"])
            col3.metric("Confidence", f"{meta['confidence']:.0%}")
            col4.metric("Latency",    f"{total_ms}ms")

            # Store run log for dashboard
            st.session_state.run_logs.append({
                "query":      prompt,
                "route":      state["route"],
                "category":   state["category"],
                "confidence": state["confidence"],
                "latency_ms": total_ms,
                "safe":       state["input_safe"],
            })

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "meta": meta,
        })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PIPELINE HEALTH DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Pipeline Health":
    st.title("📊 Pipeline Health Dashboard")

    # ── Vector store stats ────────────────────────────────────────────────────
    st.subheader("Vector Store")
    try:
        from src.rag.vectorstore import VectorStore
        vs = VectorStore()
        vstats = vs.get_stats()
        col1, col2 = st.columns(2)
        col1.metric("Chunks in Vector Store", vstats["total_chunks"])
        col2.metric("Collection", vstats["collection_name"])
    except Exception as e:
        st.error(f"Vector store error: {e}")

    st.markdown("---")

    # ── ML model stats ────────────────────────────────────────────────────────
    st.subheader("ML Classifier")
    try:
        import json
        meta_path = Path("data/model/metadata.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            col1, col2 = st.columns(2)
            col1.metric("Best Model",  meta.get("best_model", "—"))
            col2.metric("Accuracy",    f"{meta.get('accuracy', 0):.1%}")
        else:
            st.warning("No model metadata found. Run training first.")
    except Exception as e:
        st.error(f"Model metadata error: {e}")

    st.markdown("---")

    # ── Session run stats ─────────────────────────────────────────────────────
    st.subheader("Session Stats")
    logs = st.session_state.get("run_logs", [])
    if not logs:
        st.info("No queries yet. Go to the Chat page and ask something!")
    else:
        total    = len(logs)
        avg_lat  = sum(r["latency_ms"] for r in logs) / total
        safe_pct = sum(1 for r in logs if r["safe"]) / total

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries",    total)
        col2.metric("Avg Latency",      f"{avg_lat:.0f}ms")
        col3.metric("Safe Input Rate",  f"{safe_pct:.0%}")
        col4.metric("Escalation Rate",  f"{sum(1 for r in logs if r['route']=='escalate')/total:.0%}")

        # Route distribution
        st.markdown("**Route Distribution**")
        routes = {}
        for r in logs:
            routes[r["route"]] = routes.get(r["route"], 0) + 1
        for route, count in routes.items():
            st.progress(count / total, text=f"{route}: {count} queries")

        # Category distribution
        st.markdown("**Category Distribution**")
        cats = {}
        for r in logs:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        for cat, count in cats.items():
            st.progress(count / total, text=f"{cat}: {count} queries")

        # Query log
        st.markdown("**Recent Queries**")
        for r in reversed(logs[-5:]):
            st.markdown(
                f"- `{r['route'].upper()}` | {r['category']} ({r['confidence']:.0%}) "
                f"| {r['latency_ms']}ms | {r['query'][:60]}"
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ADMIN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️ Admin":
    st.title("🛠️ Admin Panel")

    # ── Re-run pipeline ───────────────────────────────────────────────────────
    st.subheader("Pipeline Controls")
    col1, col2, col3 = st.columns(3)

    if col1.button("▶ Run Data Pipeline"):
        with st.spinner("Running pipeline..."):
            import subprocess
            result = subprocess.run(
                ["python", "scripts/run_pipeline.py"],
                capture_output=True, text=True, cwd="."
            )
            st.code(result.stdout or result.stderr)

    if col2.button("▶ Rebuild Vector Store"):
        with st.spinner("Rebuilding vector store..."):
            import subprocess
            result = subprocess.run(
                ["python", "scripts/build_vectorstore.py"],
                capture_output=True, text=True, cwd="."
            )
            st.code(result.stdout or result.stderr)

    if col3.button("▶ Retrain Classifier"):
        with st.spinner("Training classifier..."):
            import subprocess
            result = subprocess.run(
                ["python", "src/ml/train.py"],
                capture_output=True, text=True, cwd="."
            )
            st.code(result.stdout or result.stderr)

    st.markdown("---")

    # ── MLflow runs ───────────────────────────────────────────────────────────
    st.subheader("MLflow Experiment Runs")
    try:
        import mlflow
        mlflow.set_experiment("ticket-classifier")
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("ticket-classifier")
        runs   = client.search_runs(exp.experiment_id)
        for r in runs[:5]:
            st.markdown(
                f"- `{r.info.run_id[:8]}` | "
                f"model={r.data.params.get('model_type','?')} | "
                f"accuracy={r.data.metrics.get('accuracy',0):.3f} | "
                f"f1={r.data.metrics.get('weighted_f1',0):.3f}"
            )
    except Exception as e:
        st.warning(f"MLflow not available: {e}")

    st.markdown("---")

    # ── Log viewer ────────────────────────────────────────────────────────────
    st.subheader("Log Viewer")
    log_path = Path("logs/app.log")
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        st.code("\n".join(lines[-30:]), language="text")
    else:
        st.info("No logs found yet.")
