import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import SupportGraph

EVAL_QUERIES = [
    ("I was charged twice this month, please refund.",     "billing"),
    ("The app won't load on my browser.",                  "technical"),
    ("How do I delete my account?",                        "general"),
    ("Our entire system is down, need help urgently!",     "urgent"),
    ("Can I get an invoice for my last payment?",          "billing"),
    ("I forgot my password and reset email never arrived.","technical"),
    ("Do you offer a student discount?",                   "general"),
    ("We lost all project data, this is critical.",        "urgent"),
    ("How do I change my username?",                       "general"),
    ("The export feature is not working for me.",          "technical"),
]

def print_result(state: dict, verbose: bool = True):
    print(f"\n  run_id:   {state['run_id']}")
    print(f"  input:    {state['user_input'][:60]}")
    print(f"  route:    {state['route']}")
    print(f"  category: {state['category']} (confidence: {state['confidence']:.0%})")
    print(f"  answer:   {state['final_answer'][:120]}")
    print(f"  latency:  {state['latencies']}")
    if not state['input_safe']:
        print(f"  blocked:  {state['input_block_reason']}")

def run_eval(graph: SupportGraph, mock: bool = True):
    print("\n── Phase 5: Agent Evaluation ─────────────────────────────────")
    correct = 0
    for query, expected in EVAL_QUERIES:
        state = graph.invoke(query, mock=mock)
        got = state["category"]
        ok = "✅" if got == expected or state["route"] == "escalate" else "⚠️"
        if ok == "✅":
            correct += 1
        print(f"  {ok} [{expected:<10}] {query[:50]}")
        print(f"      → route={state['route']}  category={got}  conf={state['confidence']:.0%}")
    print(f"\n  Result: {correct}/{len(EVAL_QUERIES)} correct")
    print(f"  ✓ Evaluation complete!\n")

def run_single(graph: SupportGraph, query: str, mock: bool = True):
    print(f"\n── Single Query ──────────────────────────────────────────────")
    state = graph.invoke(query, mock=mock)
    print_result(state)

def run_interactive(graph: SupportGraph, mock: bool = True):
    print("\n── Interactive Mode (type 'quit' to exit) ────────────────────")
    while True:
        query = input("\n  You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        state = graph.invoke(query, mock=mock)
        print(f"  Bot: {state['final_answer']}")
        print(f"  [route={state['route']} | category={state['category']} | conf={state['confidence']:.0%}]")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",      action="store_true", help="Run evaluation on 10 queries")
    parser.add_argument("--query",     type=str,            help="Run a single query")
    parser.add_argument("--diagram",   action="store_true", help="Show graph structure")
    parser.add_argument("--mock",      action="store_true", default=True, help="Use mock LLM")
    args = parser.parse_args()

    graph = SupportGraph()

    if args.diagram:
        graph.diagram()
    elif args.eval:
        run_eval(graph, mock=args.mock)
    elif args.query:
        run_single(graph, args.query, mock=args.mock)
    else:
        run_interactive(graph, mock=args.mock)

if __name__ == "__main__":
    main()
