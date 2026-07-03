import os
import time
import json

MOCK_RESPONSES = {
    "billing":   "Based on our billing policy, refunds are processed within 5-7 business days. Please contact support with your order ID.",
    "technical": "Try clearing your browser cache and cookies. If the issue persists, try a different browser or reset your password.",
    "general":   "You can manage your account settings by going to Settings > Account. For further help, contact our support team.",
    "urgent":    "This sounds urgent. I'm escalating your case to our senior support team immediately. You will be contacted within 1 hour.",
}

def call_llm(prompt: str, mock: bool = True) -> tuple[str, dict]:
    """
    Call an LLM with the given prompt.
    Uses mock mode by default — swap mock=False for real API calls.
    Returns (response_text, stats).
    """
    start = time.time()

    if mock:
        # Detect category from prompt keywords for realistic mock
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["refund", "charge", "billing", "payment", "invoice"]):
            response = MOCK_RESPONSES["billing"]
        elif any(w in prompt_lower for w in ["login", "password", "slow", "error", "crash", "app"]):
            response = MOCK_RESPONSES["technical"]
        elif any(w in prompt_lower for w in ["urgent", "down", "critical", "breach", "lost"]):
            response = MOCK_RESPONSES["urgent"]
        else:
            response = MOCK_RESPONSES["general"]

        elapsed = round((time.time() - start) * 1000, 2)
        return response, {"mode": "mock", "latency_ms": elapsed}

    # Real API call using Anthropic
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        response = message.content[0].text
        elapsed = round((time.time() - start) * 1000, 2)
        return response, {"mode": "anthropic", "latency_ms": elapsed}

    except Exception as e:
        elapsed = round((time.time() - start) * 1000, 2)
        return f"LLM error: {e}", {"mode": "error", "latency_ms": elapsed}
