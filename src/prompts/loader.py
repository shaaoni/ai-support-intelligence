# src/prompts/loader.py
import re
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"

def get_prompt(name: str, version: str = None, **kwargs) -> str:
    """
    Load a prompt template, fill variables, and return the final string.
    
    Args:
        name:    template name without extension e.g. 'rag_answer'
        version: optional e.g. 'v1.0' — raises error if file version doesn't match
        **kwargs: variables to fill into the template e.g. question="...", context="..."
    
    Returns:
        Filled prompt string ready to send to an LLM
    """
    path = TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    raw = path.read_text(encoding="utf-8")

    # Extract version from header comment
    file_version = _extract_version(raw)

    # Check version if requested
    if version and file_version and version != file_version:
        raise ValueError(f"Version mismatch for '{name}': wanted {version}, got {file_version}")

    # Strip header comment lines (lines starting with #)
    lines = [l for l in raw.splitlines() if not l.startswith("#")]
    template = "\n".join(lines).strip()

    # Find required variables
    required = set(re.findall(r"\{(\w+)\}", template))
    missing = required - set(kwargs.keys())
    if missing:
        raise ValueError(f"Missing variables for '{name}': {missing}")

    # Fill template
    filled = template.format(**kwargs)
    return filled


def get_prompt_metadata(name: str) -> dict:
    """Return version and purpose metadata from a template header."""
    path = TEMPLATES_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    raw = path.read_text(encoding="utf-8")
    return {
        "name": name,
        "version": _extract_version(raw),
        "purpose": _extract_purpose(raw),
        "path": str(path),
    }


def list_prompts() -> list[dict]:
    """List all available prompt templates with metadata."""
    prompts = []
    for p in sorted(TEMPLATES_DIR.glob("*.txt")):
        prompts.append(get_prompt_metadata(p.stem))
    return prompts


def _extract_version(raw: str) -> str:
    match = re.search(r"#.*?(v\d+\.\d+)", raw)
    return match.group(1) if match else None


def _extract_purpose(raw: str) -> str:
    match = re.search(r"purpose:\s*(.+)", raw)
    return match.group(1).strip() if match else None
