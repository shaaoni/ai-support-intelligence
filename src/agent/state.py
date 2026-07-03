from typing import TypedDict, Optional

class AgentState(TypedDict):
    # Input
    run_id: str
    user_input: str
    mock: bool

    # Guardrails
    input_safe: bool
    input_block_reason: str

    # Classification
    category: str
    confidence: float

    # Retrieval
    retrieved_chunks: list[dict]

    # LLM
    prompt: str
    llm_response: str
    output_safe: bool

    # Routing
    route: str          # "rag" | "escalate" | "blocked"
    final_answer: str

    # Observability
    latencies: dict     # node_name -> ms
