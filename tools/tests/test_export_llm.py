"""Verify tools/export_llm.py covers all production modules."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def test_export_order_covers_all_production_modules():
    """Every .py file in lab/ and tools/ (excluding tests and __init__)
    must appear in export_llm.ORDER."""
    from tools.export_llm import ORDER

    exported = set(ORDER)

    for subdir in ["lab", "tools"]:
        for p in (ROOT / subdir).glob("*.py"):
            if p.name.startswith("__"):
                continue
            rel = f"{subdir}/{p.name}"
            # Skip the export script itself and known non-production files
            if p.name in ("export_llm.py", "validate_phase4.py"):
                continue
            assert rel in exported, f"{rel} missing from export_llm.ORDER"


def test_export_script_runs_without_error():
    """The export script must produce output without crashing."""
    import subprocess
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "export_llm.py")],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, f"export_llm.py failed: {result.stderr}"
    assert "lab/market.py" in result.stdout
    assert "lab/evaluate.py" in result.stdout
