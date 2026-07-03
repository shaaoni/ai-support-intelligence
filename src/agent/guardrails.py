import re
import time

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"jailbreak",
    r"system prompt",
    r"you are now",
    r"forget your instructions",
    r"disregard your",
    r"bypass your",
    r"act as if",
    r"pretend you are",
]

BLOCKED_OUTPUT_PHRASES = [
    "my instructions are",
    "my system prompt",
    "i am programmed to",
    "as an ai language model",
]

FALLBACK_INPUT  = "I'm sorry, I can't process that request. Please describe your support issue."
FALLBACK_OUTPUT = "I'm sorry, I wasn't able to generate a safe response. Please contact support directly."

def check_input(text: str) -> tuple[bool, str]:
    """
    Check user input for prompt injection.
    Returns (is_safe, reason).
    """
    start = time.time()
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            elapsed = round((time.time() - start) * 1000, 2)
            return False, f"Blocked pattern detected: '{pattern}' ({elapsed}ms)"
    return True, "ok"

def check_output(text: str) -> tuple[bool, str]:
    """
    Check LLM output for blocked phrases.
    Returns (is_safe, reason).
    """
    start = time.time()
    lowered = text.lower()
    for phrase in BLOCKED_OUTPUT_PHRASES:
        if phrase in lowered:
            elapsed = round((time.time() - start) * 1000, 2)
            return False, f"Blocked output phrase: '{phrase}' ({elapsed}ms)"
    return True, "ok"

def safe_input(text: str) -> str:
    """Return original text if safe, fallback message if not."""
    is_safe, _ = check_input(text)
    return text if is_safe else FALLBACK_INPUT

def safe_output(text: str) -> str:
    """Return original text if safe, fallback message if not."""
    is_safe, _ = check_output(text)
    return text if is_safe else FALLBACK_OUTPUT
