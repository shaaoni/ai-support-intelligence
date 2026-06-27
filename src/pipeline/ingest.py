import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"billing", "technical", "general", "urgent"}

BRAND_ACCOUNTS = {
    "sprintcare", "ask_spectrum", "tmobilhelp", "verizon",
    "xfinity", "amazonhelp", "appleSupport", "delta", "united"
}

def _make_record(id, text, category, timestamp, source, metadata=None):
    """Standard schema for every record regardless of source."""
    return {
        "id": id,
        "text": text,
        "category": category,
        "timestamp": timestamp,
        "source": source,
        "metadata": metadata or {},
    }

def _infer_category(text):
    """Infer category from tweet text keywords."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["bill", "charge", "payment", "refund", "invoice", "price"]):
        return "billing"
    if any(w in text_lower for w in ["down", "error", "crash", "slow", "broken", "fix", "issue", "problem"]):
        return "technical"
    if any(w in text_lower for w in ["urgent", "emergency", "critical", "asap", "immediately"]):
        return "urgent"
    return "general"

def load_csv_tickets(path="data/raw/CSVs/tickets.csv"):
    """Load support tickets from a CSV file (synthetic or Kaggle format)."""
    records = []
    path = Path(path)

    if not path.exists():
        logger.warning(f"CSV file not found: {path}")
        return records

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            # Handle Kaggle twcs format
            if "tweet_id" in fieldnames:
                # Only load inbound (customer) tweets, skip brand responses
                if row.get("inbound", "").strip().lower() != "true":
                    continue
                text = row.get("text", "").strip()
                record = _make_record(
                    id=f"TWT-{row.get('tweet_id', '').strip()}",
                    text=text,
                    category=_infer_category(text),
                    timestamp=row.get("created_at", datetime.now().isoformat()),
                    source="csv",
                    metadata={"author_id": row.get("author_id", "")},
                )
            else:
                # Handle synthetic tickets format
                record = _make_record(
                    id=row.get("id", "").strip(),
                    text=row.get("text", "").strip(),
                    category=row.get("category", "").strip(),
                    timestamp=row.get("timestamp", "").strip(),
                    source="csv",
                    metadata={"original_row": dict(row)},
                )
            records.append(record)

    logger.info(f"Loaded {len(records)} tickets from {path}")
    return records

def load_kaggle_tweets(path="data/raw/CSVs/twcs/twcs.csv", max_records=2000):
    """Load real customer tweets from the Kaggle Twitter dataset."""
    records = []
    path = Path(path)

    if not path.exists():
        logger.warning(f"Kaggle dataset not found: {path}")
        return records

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_records:
                break
            # Only inbound (customer) tweets
            if row.get("inbound", "").strip().lower() != "true":
                continue
            text = row.get("text", "").strip()
            if not text:
                continue
            record = _make_record(
                id=f"TWT-{row.get('tweet_id', i)}",
                text=text,
                category=_infer_category(text),
                timestamp=row.get("created_at", datetime.now().isoformat()),
                source="kaggle_twitter",
                metadata={"author_id": row.get("author_id", "")},
            )
            records.append(record)

    logger.info(f"Loaded {len(records)} Kaggle tweets from {path}")
    return records

def load_markdown_docs(directory="data/raw/docs"):
    """Load markdown documents from a directory."""
    records = []
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Docs directory not found: {directory}")
        return records

    for md_file in directory.glob("*.md"):
        text = md_file.read_text(encoding="utf-8").strip()
        record = _make_record(
            id=md_file.stem,
            text=text,
            category="general",
            timestamp=datetime.now().isoformat(),
            source="markdown",
            metadata={"filename": md_file.name},
        )
        records.append(record)

    logger.info(f"Loaded {len(records)} markdown docs from {directory}")
    return records

def load_all_sources():
    """Load and combine all data sources."""
    records = []
    records.extend(load_csv_tickets())
    records.extend(load_kaggle_tweets())
    records.extend(load_markdown_docs())
    logger.info(f"Total records loaded: {len(records)}")
    return records
