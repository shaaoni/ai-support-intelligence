import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

CATEGORIES = ["billing", "technical", "general", "urgent"]

TEXTS = {
    "billing": [
        "I was charged twice this month, please refund the duplicate charge.",
        "How do I update my credit card details?",
        "I need an invoice for my last payment.",
        "My payment failed but I was still charged.",
        "Can I switch from monthly to annual billing?",
    ],
    "technical": [
        "The app won't load on my browser.",
        "I forgot my password and the reset email never arrived.",
        "The export feature is not working for me.",
        "App is very slow since the last update.",
        "I cannot log in with my Google account.",
    ],
    "general": [
        "How do I delete my account?",
        "Is there a mobile app available?",
        "What are your support hours?",
        "How do I change my username?",
        "Do you offer a student discount?",
    ],
    "urgent": [
        "My entire team is locked out of the system urgently.",
        "We have lost all our project data, this is critical.",
        "Security breach suspected on our account.",
        "Production system is down, need immediate help.",
        "All users getting error 500, urgent fix needed.",
    ],
}

def generate_tickets(n_valid=50, n_invalid=5):
    tickets = []
    base_time = datetime.now() - timedelta(days=30)

    # Valid tickets
    for i in range(n_valid):
        category = random.choice(CATEGORIES)
        ticket = {
            "id": f"TKT-{1000 + i}",
            "text": random.choice(TEXTS[category]),
            "category": category,
            "timestamp": (base_time + timedelta(hours=i * 12)).isoformat(),
            "source": "csv",
        }
        tickets.append(ticket)

    # Invalid tickets (intentional — quality checks should catch these)
    invalid = [
        {"id": "", "text": "Missing ID ticket", "category": "general", "timestamp": datetime.now().isoformat(), "source": "csv"},
        {"id": "TKT-9001", "text": "", "category": "billing", "timestamp": datetime.now().isoformat(), "source": "csv"},
        {"id": "TKT-9002", "text": "hi", "category": "billing", "timestamp": datetime.now().isoformat(), "source": "csv"},
        {"id": "TKT-9003", "text": "Valid text but unknown category", "category": "unknown_cat", "timestamp": datetime.now().isoformat(), "source": "csv"},
        {"id": "TKT-9004", "text": "Future timestamp ticket", "category": "general", "timestamp": (datetime.now() + timedelta(days=10)).isoformat(), "source": "csv"},
    ]
    tickets.extend(invalid)
    random.shuffle(tickets)
    return tickets

def main():
    output_path = Path("data/raw/CSVs/tickets.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tickets = generate_tickets()
    fieldnames = ["id", "text", "category", "timestamp", "source"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)

    valid = sum(1 for t in tickets if t["id"] and t["text"] and t["category"] in CATEGORIES)
    invalid = len(tickets) - valid
    cats = {}
    for t in tickets:
        cats[t["category"]] = cats.get(t["category"], 0) + 1

    print(f"✓ Generated {len(tickets)} tickets → {output_path}")
    print(f"  Valid:   {valid}")
    print(f"  Invalid: {invalid}  (intentional — quality checks should catch these)")
    print(f"  Categories: {cats}")

if __name__ == "__main__":
    main()
