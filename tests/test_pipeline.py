import pytest
from src.pipeline.ingest import load_csv_tickets, load_markdown_docs, load_all_sources, _make_record
from src.pipeline.quality import check_completeness, check_consistency, check_timeliness, run_quality_checks
from src.pipeline.expectations import (
    expect_column_to_exist,
    expect_column_values_to_not_be_null,
    expect_column_values_to_be_in_set,
    expect_column_value_lengths_to_be_between,
    run_expectation_suite,
)
from datetime import datetime, timedelta

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_valid_record(**overrides):
    base = {
        "id": "TKT-001",
        "text": "I was charged twice this month please help.",
        "category": "billing",
        "timestamp": datetime.now().isoformat(),
        "source": "csv",
        "metadata": {},
    }
    base.update(overrides)
    return base

# ── _make_record ──────────────────────────────────────────────────────────────
def test_make_record_returns_all_fields():
    r = _make_record("1", "some text", "billing", datetime.now().isoformat(), "csv")
    assert r["id"] == "1"
    assert r["text"] == "some text"
    assert r["category"] == "billing"
    assert r["source"] == "csv"
    assert r["metadata"] == {}

def test_make_record_accepts_metadata():
    r = _make_record("1", "text", "billing", datetime.now().isoformat(), "csv", metadata={"foo": "bar"})
    assert r["metadata"]["foo"] == "bar"

# ── load_csv_tickets ──────────────────────────────────────────────────────────
def test_load_csv_tickets_returns_list():
    records = load_csv_tickets("data/raw/CSVs/tickets.csv")
    assert isinstance(records, list)

def test_load_csv_tickets_missing_file_returns_empty():
    records = load_csv_tickets("data/raw/CSVs/nonexistent.csv")
    assert records == []

def test_load_csv_tickets_have_required_fields():
    records = load_csv_tickets("data/raw/CSVs/tickets.csv")
    if records:
        r = records[0]
        assert "id" in r
        assert "text" in r
        assert "category" in r
        assert "source" in r

def test_load_csv_tickets_source_is_csv():
    records = load_csv_tickets("data/raw/CSVs/tickets.csv")
    for r in records:
        assert r["source"] == "csv"

# ── load_markdown_docs ────────────────────────────────────────────────────────
def test_load_markdown_docs_returns_list():
    records = load_markdown_docs("data/raw/docs")
    assert isinstance(records, list)

def test_load_markdown_docs_missing_dir_returns_empty():
    records = load_markdown_docs("data/raw/nonexistent")
    assert records == []

def test_load_markdown_docs_source_is_markdown():
    records = load_markdown_docs("data/raw/docs")
    for r in records:
        assert r["source"] == "markdown"

def test_load_markdown_docs_have_text():
    records = load_markdown_docs("data/raw/docs")
    for r in records:
        assert len(r["text"]) > 0

# ── check_completeness ────────────────────────────────────────────────────────
def test_completeness_passes_valid_records():
    records = [make_valid_record()]
    issues = check_completeness(records)
    assert issues == []

def test_completeness_catches_missing_id():
    records = [make_valid_record(id="")]
    issues = check_completeness(records)
    assert any(i["issue"] == "missing_id" for i in issues)

def test_completeness_catches_missing_text():
    records = [make_valid_record(text="")]
    issues = check_completeness(records)
    assert any(i["issue"] == "missing_text" for i in issues)

# ── check_consistency ─────────────────────────────────────────────────────────
def test_consistency_passes_valid_category():
    records = [make_valid_record(category="billing")]
    issues = check_consistency(records)
    assert issues == []

def test_consistency_catches_invalid_category():
    records = [make_valid_record(category="unknown_cat")]
    issues = check_consistency(records)
    assert any("invalid_category" in i["issue"] for i in issues)

def test_consistency_catches_short_text():
    records = [make_valid_record(text="hi")]
    issues = check_consistency(records)
    assert any("text_too_short" in i["issue"] for i in issues)

# ── check_timeliness ──────────────────────────────────────────────────────────
def test_timeliness_passes_past_timestamp():
    records = [make_valid_record(timestamp=(datetime.now() - timedelta(days=1)).isoformat())]
    issues = check_timeliness(records)
    assert issues == []

def test_timeliness_catches_future_timestamp():
    records = [make_valid_record(timestamp=(datetime.now() + timedelta(days=10)).isoformat())]
    issues = check_timeliness(records)
    assert any("future_timestamp" in i["issue"] for i in issues)

def test_timeliness_catches_invalid_format():
    records = [make_valid_record(timestamp="not-a-date")]
    issues = check_timeliness(records)
    assert any("invalid_timestamp_format" in i["issue"] for i in issues)

# ── run_quality_checks ────────────────────────────────────────────────────────
def test_run_quality_checks_returns_three_values():
    records = [make_valid_record()]
    result = run_quality_checks(records)
    assert len(result) == 3

def test_run_quality_checks_clean_count():
    records = [make_valid_record(), make_valid_record(id="TKT-002")]
    clean, quarantine, report = run_quality_checks(records)
    assert len(clean) == 2
    assert len(quarantine) == 0

def test_run_quality_checks_quarantine_bad_record():
    records = [make_valid_record(), make_valid_record(id="", text="")]
    clean, quarantine, report = run_quality_checks(records)
    assert len(quarantine) >= 1

# ── expectations ──────────────────────────────────────────────────────────────
def test_expect_column_to_exist_passes():
    records = [make_valid_record()]
    result = expect_column_to_exist(records, "id")
    assert result["passed"] is True

def test_expect_column_to_exist_fails():
    records = [{"text": "no id here"}]
    result = expect_column_to_exist(records, "id")
    assert result["passed"] is False

def test_expect_not_null_passes():
    records = [make_valid_record()]
    result = expect_column_values_to_not_be_null(records, "text")
    assert result["passed"] is True

def test_expect_not_null_fails():
    records = [make_valid_record(text="")]
    result = expect_column_values_to_not_be_null(records, "text")
    assert result["passed"] is False

def test_expect_in_set_passes():
    records = [make_valid_record(category="billing")]
    result = expect_column_values_to_be_in_set(records, "category", {"billing", "technical"})
    assert result["passed"] is True

def test_expect_in_set_fails():
    records = [make_valid_record(category="unknown")]
    result = expect_column_values_to_be_in_set(records, "category", {"billing", "technical"})
    assert result["passed"] is False

def test_expect_length_passes():
    records = [make_valid_record(text="This is a valid long enough text.")]
    result = expect_column_value_lengths_to_be_between(records, "text", 10, 5000)
    assert result["passed"] is True

def test_expect_length_fails():
    records = [make_valid_record(text="hi")]
    result = expect_column_value_lengths_to_be_between(records, "text", 10, 5000)
    assert result["passed"] is False
