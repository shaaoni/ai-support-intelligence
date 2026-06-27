import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"billing", "technical", "general", "urgent"}

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

def load_csv_tickets(path="data/raw/CSVs/tickets.csv"):
    """Load support tickets from a CSV file."""
    records = []
    path = Path(path)

    if not path.exists():
        logger.warning(f"CSV file not found: {path}")
        return records

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
    records.extend(load_markdown_docs())
    logger.info(f"Total records loaded: {len(records)}")
    return records
