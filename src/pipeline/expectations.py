import logging
from datetime import datetime

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"billing", "technical", "general", "urgent"}


def expect_column_to_exist(records, column):
    """Check that a key exists in every record."""
    missing = [i for i, r in enumerate(records) if column not in r]
    passed = len(missing) == 0
    return {
        "expectation": f"expect_column_to_exist:{column}",
        "passed": passed,
        "failed_rows": missing,
    }


def expect_column_values_to_not_be_null(records, column):
    """Check that a key is never null or empty."""
    failed = [r.get("id", i) for i, r in enumerate(records) if not r.get(column)]
    passed = len(failed) == 0
    return {
        "expectation": f"expect_column_values_to_not_be_null:{column}",
        "passed": passed,
        "failed_ids": failed,
    }


def expect_column_values_to_be_in_set(records, column, value_set):
    """Check that all values in a column belong to an allowed set."""
    failed = [r.get("id", i) for i, r in enumerate(records)
              if r.get(column) not in value_set]
    passed = len(failed) == 0
    return {
        "expectation": f"expect_column_values_to_be_in_set:{column}",
        "passed": passed,
        "failed_ids": failed,
    }


def expect_column_value_lengths_to_be_between(records, column, min_len=10, max_len=5000):
    """Check that text length is within bounds."""
    failed = []
    for r in records:
        val = r.get(column, "")
        if val and not (min_len <= len(val) <= max_len):
            failed.append(r.get("id"))
    passed = len(failed) == 0
    return {
        "expectation": f"expect_column_value_lengths_to_be_between:{column}:{min_len}-{max_len}",
        "passed": passed,
        "failed_ids": failed,
    }


def run_expectation_suite(records):
    """Run all expectations and return a suite report."""
    print("\n📋  Running Great Expectations suite...")

    suite = [
        expect_column_to_exist(records, "id"),
        expect_column_to_exist(records, "text"),
        expect_column_to_exist(records, "category"),
        expect_column_to_exist(records, "timestamp"),
        expect_column_values_to_not_be_null(records, "id"),
        expect_column_values_to_not_be_null(records, "text"),
        expect_column_values_to_be_in_set(records, "category", VALID_CATEGORIES),
        expect_column_value_lengths_to_be_between(records, "text", min_len=10, max_len=5000),
    ]

    passed = sum(1 for e in suite if e["passed"])
    failed = len(suite) - passed

    for e in suite:
        status = "✓" if e["passed"] else "✗"
        print(f"    {status}  {e['expectation']}")

    print(f"\n    Expectations passed: {passed}/{len(suite)}")

    return {
        "total": len(suite),
        "passed": passed,
        "failed": failed,
        "results": suite,
        "generated_at": datetime.now().isoformat(),
    }
