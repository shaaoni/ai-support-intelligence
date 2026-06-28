import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.prompts.loader import get_prompt, list_prompts

TEST_TICKETS = [
    ("I was charged twice this month.",           "billing"),
    ("Can I get an invoice for my last payment?", "billing"),
    ("The app won't load on Chrome.",             "technical"),
    ("I keep getting a 403 error on reports.",    "technical"),
    ("How many users can I add to Pro plan?",     "general"),
    ("Is my data backed up automatically?",       "general"),
    ("URGENT our whole team is locked out now.",  "urgent"),
    ("Critical security breach suspected.",       "urgent"),
    ("How do I cancel my subscription?",          "billing"),
    ("The mobile app crashes every time.",        "technical"),
]

FEW_SHOT_EXAMPLES = """Examples:
Ticket: I was double charged last week → billing
Ticket: App crashes on startup → technical
Ticket: How do I add a team member? → general
Ticket: All our data has been deleted → urgent

"""

def test_zero_shot_vs_few_shot():
    print("\n── Zero-shot vs Few-shot Classifier Prompts ─────────────────")
    print(f"  {'Ticket':<48} {'Expected':<12} {'Zero-shot prompt ends':<25} {'Few-shot prompt ends'}")
    print(f"  {'─'*48}  {'─'*10}  {'─'*23}  {'─'*20}")

    for ticket, expected in TEST_TICKETS:
        zero = get_prompt("classifier_label", ticket_text=ticket, examples="")
        few  = get_prompt("classifier_label", ticket_text=ticket, examples=FEW_SHOT_EXAMPLES)

        zero_end = zero.strip().splitlines()[-1]
        few_end  = few.strip().splitlines()[-1]

        print(f"  {ticket[:46]:<48} {expected:<12} {zero_end:<25} {few_end}")

    print(f"\n  Key difference:")
    print(f"  Zero-shot: no examples → model relies on category descriptions only")
    print(f"  Few-shot:  4 examples  → model sees pattern before classifying")

def test_query_rewrite():
    print("\n── Query Rewrite Prompts ─────────────────────────────────────")
    messy_queries = [
        "HELP!! my app wont load at all!!!",
        "hi there, um, i think maybe my payment didnt go through??",
        "how do i do the export thing for csv files",
        "urgent urgent urgent team locked out please help asap",
    ]
    for q in messy_queries:
        prompt = get_prompt("query_rewrite", message=q)
        last_line = prompt.strip().splitlines()[-1]
        print(f"  Input:  {q[:60]}")
        print(f"  Prompt ends with: '{last_line}'")
        print()

def test_rag_answer():
    print("\n── RAG Answer Prompt ─────────────────────────────────────────")
    prompt = get_prompt(
        "rag_answer",
        context="To cancel your subscription, go to Settings > Subscription > Cancel Plan.",
        question="How do I cancel my subscription?",
    )
    print(prompt)

def test_all_templates():
    print("\n── All Available Templates ───────────────────────────────────")
    for p in list_prompts():
        print(f"  ✓ {p['name']:25} version={p['version']}  purpose={p['purpose']}")

if __name__ == "__main__":
    test_all_templates()
    test_zero_shot_vs_few_shot()
    test_query_rewrite()
    test_rag_answer()
    print("\n  ✓ Phase 4 prompt engineering tests complete!\n")
