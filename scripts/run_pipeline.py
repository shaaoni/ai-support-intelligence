import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.pipeline.ingest import load_all_sources
from src.pipeline.quality import run_quality_checks

# ── Logging setup ────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("run_pipeline")


def print_section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main(verbose=False, save_report=False):
    print_section("Phase 1 — Data Ingestion & Quality Pipeline")

    # ── Step 1: Ingest ───────────────────────────────────────────────────────
    print("\n📥  Loading all sources...")
    records = load_all_sources()
    print(f"    ✓ Total records loaded: {len(records)}")

    if verbose:
        sources = {}
        for r in records:
            sources[r["source"]] = sources.get(r["source"], 0) + 1
        for src, count in sources.items():
            print(f"       • {src}: {count} records")

    # ── Step 2: Quality checks ───────────────────────────────────────────────
    print("\n🔍  Running quality checks...")
    clean, quarantine, report = run_quality_checks(records)
    print(f"    ✓ Clean records:      {report['clean']}")
    print(f"    ✗ Quarantined:        {report['quarantined']}")

    if verbose and quarantine:
        print("\n    Quarantined records:")
        for issue in report["completeness_issues"] + report["consistency_issues"] + report["timeliness_issues"]:
            print(f"       • ID={issue['id']} → {issue['issue']}")

    # ── Step 3: Summary ──────────────────────────────────────────────────────
    print_section("Summary")
    print(f"  Total loaded:     {report['total']}")
    print(f"  Clean:            {report['clean']}")
    print(f"  Quarantined:      {report['quarantined']}")
    print(f"  Completeness ✗:   {len(report['completeness_issues'])}")
    print(f"  Consistency  ✗:   {len(report['consistency_issues'])}")
    print(f"  Timeliness   ✗:   {len(report['timeliness_issues'])}")

    # ── Step 4: Save report ──────────────────────────────────────────────────
    if save_report:
        report_path = Path("data/expectations/pipeline_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report["generated_at"] = datetime.now().isoformat()
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  📄 Report saved → {report_path}")

    print("\n  ✓ Phase 1 complete!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    args = parser.parse_args()
    main(verbose=args.verbose, save_report=args.save_report)
