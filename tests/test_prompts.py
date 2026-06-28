import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompts.loader import get_prompt, get_prompt_metadata, list_prompts

# ── list_prompts ──────────────────────────────────────────────────────────────

def test_list_prompts_returns_all_templates():
    prompts = list_prompts()
    names = [p["name"] for p in prompts]
    for required in ["rag_answer", "classifier_label", "query_rewrite", "fallback", "system_base", "escalation"]:
        assert required in names, f"Missing template: {required}"

def test_list_prompts_have_version():
    for p in list_prompts():
        assert p["version"] is not None, f"Missing version in {p['name']}"

def test_list_prompts_have_purpose():
    for p in list_prompts():
        assert p["purpose"] is not None, f"Missing purpose in {p['name']}"

# ── get_prompt_metadata ───────────────────────────────────────────────────────

def test_metadata_returns_correct_name():
    meta = get_prompt_metadata("rag_answer")
    assert meta["name"] == "rag_answer"

def test_metadata_raises_for_missing_template():
    with pytest.raises(FileNotFoundError):
        get_prompt_metadata("nonexistent_template")

# ── get_prompt ────────────────────────────────────────────────────────────────

def test_rag_answer_fills_correctly():
    prompt = get_prompt("rag_answer", context="Some context here.", question="What is this?")
    assert "Some context here." in prompt
    assert "What is this?" in prompt

def test_query_rewrite_fills_correctly():
    prompt = get_prompt("query_rewrite", message="my app wont load help!!")
    assert "my app wont load help!!" in prompt

def test_classifier_label_zero_shot():
    prompt = get_prompt("classifier_label", ticket_text="I was charged twice.", examples="")
    assert "I was charged twice." in prompt
    assert "billing" in prompt

def test_classifier_label_few_shot():
    examples = "Ticket: App crashes → technical\n"
    prompt = get_prompt("classifier_label", ticket_text="I was charged twice.", examples=examples)
    assert "App crashes" in prompt
    assert "I was charged twice." in prompt

def test_fallback_fills_correctly():
    prompt = get_prompt("fallback", question="What is the meaning of life?")
    assert "What is the meaning of life?" in prompt

def test_escalation_fills_correctly():
    prompt = get_prompt("escalation", category="urgent", message="Team locked out!")
    assert "urgent" in prompt
    assert "Team locked out!" in prompt

def test_missing_variable_raises_error():
    with pytest.raises(ValueError, match="Missing variables"):
        get_prompt("rag_answer", context="Some context.")  # missing question

def test_version_mismatch_raises_error():
    with pytest.raises(ValueError, match="Version mismatch"):
        get_prompt("rag_answer", version="v9.9", context="ctx", question="q")

def test_missing_template_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_prompt("nonexistent_template")

def test_system_base_loads():
    meta = get_prompt_metadata("system_base")
    assert meta["version"] is not None
