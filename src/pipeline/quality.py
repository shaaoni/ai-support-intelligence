import logging
from datetime import datetime

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {"billing", "technical", "general", "urgent"}
MIN_TEXT_LENGTH = 10


def check_completeness(records):
    """Check that required fields are not empty."""
    issues = []
    for r in records:
        if not r.get("id"):
            issues.append({"id": r.get("id"), "issue": "missing_id"})
        if not r.get("text"):
            issues.append({"id": r.get("id"), "issue": "missing_text"})
    return issues


def check_consistency(records):
    """Check categories are valid and text is long enough."""
    issues = []
    for r in records:
        if r.get("category") not in VALID_CATEGORIES:
            issues.append({"id": r.get("id"), "issue": f"invalid_category: {r.get('category')}"})
        if r.get("text") and len(r.get("text", "")) < MIN_TEXT_LENGTH:
            issues.append({"id": r.get("id"), "issue": f"text_too_short: {len(r.get('text', ''))} chars"})
    return issues


def check_timeliness(records):
    """Check that timestamps are not in the future."""
    issues = []
    now = datetime.now()
    for r in records:
        ts = r.get("timestamp", "")
        if not ts:
            continue
        try:
            record_time = datetime.fromisoformat(ts)
            if record_time > now:
                issues.append({"id": r.get("id"), "issue": f"future_timestamp: {ts}"})
        except ValueError:
            issues.append({"id": r.get("id"), "issue": f"invalid_timestamp_format: {ts}"})
    return issues


def run_quality_checks(records):
    """Run all quality checks and return a report."""
    completeness_issues = check_completeness(records)
    consistency_issues  = check_consistency(records)
    timeliness_issues   = check_timeliness(records)

    all_issues = completeness_issues + consistency_issues + timeliness_issues
    bad_ids    = {i["id"] for i in all_issues}

    clean     = [r for r in records if r.get("id") not in bad_ids]
    quarantine = [r for r in records if r.get("id") in bad_ids]

    report = {
        "total":              len(records),
        "clean":              len(clean),
        "quarantined":        len(quarantine),
        "completeness_issues": completeness_issues,
        "consistency_issues":  consistency_issues,
        "timeliness_issues":   timeliness_issues,
    }

    logger.info(f"Quality check — total: {len(records)}, clean: {len(clean)}, quarantined: {len(quarantine)}")
    return clean, quarantine, report
