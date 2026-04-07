# Copyright 2026 Entalpic
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_import_has_no_cwd_side_effects(tmp_path: Path) -> None:
    python_src = Path(__file__).resolve().parents[1] / "python"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(python_src)

    res = subprocess.run(
        [sys.executable, "-c", "import atompack"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert res.stdout == ""
    assert res.stderr == ""
    assert list(tmp_path.iterdir()) == []
