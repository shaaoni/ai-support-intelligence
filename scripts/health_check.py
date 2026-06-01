"""
scripts/health_check.py — run this after setup to verify everything is wired.

Usage:
    python scripts/health_check.py

What it checks:
    ✓ Python version
    ✓ All required packages importable
    ✓ .env loaded and keys present
    ✓ Data directories exist
    ✓ Logger works
"""
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "✓" if ok else "✗"
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    suffix = f"  {detail}" if detail else ""
    print(f"  {color}{status}{reset}  {label}{suffix}")
    return ok


def run():
    print("\n── Environment health check ──────────────────────────────\n")
    all_ok = True

    # Python version
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 10
    all_ok &= check(f"Python version", ok, f"(found {major}.{minor}, need 3.10+)")

    # Required packages
    packages = [
        ("dotenv",              "python-dotenv"),
        ("pandas",              "pandas"),
        ("pathlib",             "built-in"),
        ("logging",             "built-in"),
    ]
    for module, pkg in packages:
        try:
            __import__(module)
            all_ok &= check(f"import {module}", True, f"({pkg})")
        except ImportError:
            all_ok &= check(f"import {module}", False, f"→ run: pip install {pkg}")

    # Optional packages (warn but don't fail)
    print()
    optional = [
        ("sklearn",             "scikit-learn"),
        ("mlflow",              "mlflow"),
        ("chromadb",            "chromadb"),
        ("langchain",           "langchain"),
        ("langgraph",           "langgraph"),
        ("streamlit",           "streamlit"),
        ("fastapi",             "fastapi"),
        ("sentence_transformers","sentence-transformers"),
    ]
    for module, pkg in optional:
        try:
            __import__(module)
            check(f"import {module}", True, f"({pkg})")
        except ImportError:
            check(f"import {module}", False, f"→ not yet installed (install in Phase {_phase(pkg)})")

    # Config / .env
    print()
    try:
        from src.config import settings
        missing = settings.validate()
        if missing:
            for m in missing:
                all_ok &= check(f".env: {m}", False, "→ add to your .env file")
        else:
            check(".env keys loaded", True)
    except Exception as e:
        all_ok &= check(".env / config", False, str(e))

    # Directories
    print()
    dirs = [
        "data/raw", "data/raw/docs", "data/chroma",
        "data/expectations", "scripts", "docs", "tests",
        "src/pipeline", "src/rag", "src/ml", "src/agent", "src/prompts",
    ]
    root = Path(__file__).resolve().parent.parent
    for d in dirs:
        exists = (root / d).is_dir()
        all_ok &= check(f"dir: {d}/", exists)

    # Logger
    print()
    try:
        from src.logger import get_logger
        log = get_logger("health_check")
        log.info("Logger test — you should see this line")
        check("Logger initialised", True, "(check logs/app.log)")
    except Exception as e:
        all_ok &= check("Logger", False, str(e))

    # Summary
    print()
    if all_ok:
        print("  \033[92m✓ All checks passed. You're ready to start Phase 1.\033[0m\n")
    else:
        print("  \033[91m✗ Some checks failed. Fix the items above before continuing.\033[0m\n")


def _phase(pkg: str) -> str:
    mapping = {
        "scikit-learn": "2", "mlflow": "2", "fastapi": "2",
        "chromadb": "3", "sentence-transformers": "3",
        "langchain": "4/5", "langgraph": "5", "streamlit": "6",
    }
    return mapping.get(pkg, "?")


if __name__ == "__main__":
    run()