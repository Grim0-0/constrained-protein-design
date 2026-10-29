"""Make ``src/`` and the repo root importable so a fresh clone can run ``pytest``.

Installing the package with ``pip install -e .`` also works and is what the
README recommends; this just removes it as a precondition for running the
tests.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
