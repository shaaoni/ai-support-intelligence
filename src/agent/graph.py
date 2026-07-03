import time
import uuid
import logging

from src.agent.state import AgentState
from src.agent.guardrails import check_input, check_output, FALLBACK_OUTPUT
from src.agent.llm import call_llm
from src.prompts.loader import get_prompt

logger = logging.getLogger(__name__)

CONFIDENCE_ESCALATE_BELOW = 0.28

# ── Nodes ─────────────────────────────────────────────────────────────────────

def node_guardrail(state: AgentState) -> AgentState:
    start = time.time()
    is_safe, reason = check_input(state["user_input"])
    state["input_safe"] = is_safe
    state["input_block_reason"] = reason if not is_safe else ""
    state["latencies"]["guardrail"] = round((time.time() - start) * 1000, 2)
    return state

def node_classify(state: AgentState) -> AgentState:
    start = time.time()
    try:
        import src.ml.serve as serve_module
        if serve_module.model is None:
            serve_module.load_model()
        text = state["user_input"]
        prediction = serve_module.model.predict([text])[0]
        proba = serve_module.model.predict_proba([text])[0]
        confidence = float(max(proba))
        state["category"] = prediction
        state["confidence"] = confidence
    except Exception as e:
        logger.warning(f"Classifier error: {e} -- defaulting to general")
        state["category"] = "general"
        state["confidence"] = 0.0
    state["latencies"]["classify"] = round((time.time() - start) * 1000, 2)
    return state

def node_retrieve(state: AgentState) -> AgentState:
    start = time.time()
    try:
        from src.rag.vectorstore import VectorStore
        vs = VectorStore()
        hits, _ = vs.query(state["user_input"], n=3)
        state["retrieved_chunks"] = hits
    except Exception as e:
        logger.warning(f"Retrieval error: {e}")
        state["retrieved_chunks"] = []
    state["latencies"]["retrieve"] = round((time.time() - start) * 1000, 2)
    return state

def node_rag_answer(state: AgentState) -> AgentState:
    start = time.time()
    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(c["text"] for c in chunks) if chunks else "No context available."
    prompt = get_prompt("rag_answer", context=context, question=state["user_input"])
    state["prompt"] = prompt
    response, llm_stats = call_llm(prompt, mock=state.get("mock", True))
    is_safe, _ = check_output(response)
    state["llm_response"] = response
    state["output_safe"] = is_safe
    state["final_answer"] = response if is_safe else FALLBACK_OUTPUT
    state["route"] = "rag"
    state["latencies"]["rag_answer"] = round((time.time() - start) * 1000, 2)
    return state

def node_escalate(state: AgentState) -> AgentState:
    start = time.time()
    category = state.get("category", "unknown")
    confidence = state.get("confidence", 0.0)
    state["route"] = "escalate"
    state["final_answer"] = (
        f"Your request has been escalated to our support team. "
        f"Category: {category} (confidence: {confidence:.0%}). "
        f"A human agent will contact you within 2 hours."
    )
    state["latencies"]["escalate"] = round((time.time() - start) * 1000, 2)
    return state

def node_blocked(state: AgentState) -> AgentState:
    start = time.time()
    state["route"] = "blocked"
    state["final_answer"] = "I'm sorry, I can't process that request. Please describe your support issue."
    state["latencies"]["blocked"] = round((time.time() - start) * 1000, 2)
    return state

# ── Routing logic ──────────────────────────────────────────────────────────────

def route_after_guardrail(state: AgentState) -> str:
    return "classify" if state["input_safe"] else "blocked"

def route_after_classify(state: AgentState) -> str:
    if state["category"] == "urgent":
        return "escalate"
    if state["confidence"] < CONFIDENCE_ESCALATE_BELOW:
        return "escalate"
    return "retrieve"

# ── Graph ──────────────────────────────────────────────────────────────────────

class SupportGraph:
    def __init__(self):
        self.nodes = {
            "guardrail":  node_guardrail,
            "classify":   node_classify,
            "retrieve":   node_retrieve,
            "rag_answer": node_rag_answer,
            "escalate":   node_escalate,
            "blocked":    node_blocked,
        }

    def invoke(self, user_input: str, mock: bool = True) -> AgentState:
        state: AgentState = {
            "run_id": str(uuid.uuid4())[:8],
            "user_input": user_input,
            "mock": mock,
            "input_safe": True,
            "input_block_reason": "",
            "category": "",
            "confidence": 0.0,
            "retrieved_chunks": [],
            "prompt": "",
            "llm_response": "",
            "output_safe": True,
            "route": "",
            "final_answer": "",
            "latencies": {},
        }

        state = node_guardrail(state)
        if route_after_guardrail(state) == "blocked":
            return node_blocked(state)

        state = node_classify(state)
        if route_after_classify(state) == "escalate":
            return node_escalate(state)

        state = node_retrieve(state)
        state = node_rag_answer(state)
        return state

    def diagram(self):
        print("""
  +--------------+
  |  user_input  |
  +------+-------+
         |
  +------v-------+
  |  guardrail   |
  +------+-------+
         | safe?
    +----+----+
   YES        NO
    |          |
  +-v------+ +-v-------+
  |classify| | blocked |
  +-+------+ +---------+
    | confident?
  +-+----------+
 YES           NO / urgent
  |             |
+-v----------+ +-v--------+
|retrieve +  | | escalate |
|rag_answer  | +----------+
+------------+
        """)
