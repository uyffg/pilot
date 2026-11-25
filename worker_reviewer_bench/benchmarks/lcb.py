"""Exact stdin comparator with early exit once a failure proves task failure."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def run_solution(code, tests, timeout):
    if not code:
        return dict(status="empty", passed=False, tests_run=0)
    valid = [t for t in tests if t.get("testtype", "stdin") == "stdin"]
    if not valid:
        return dict(status="no_supported_tests", passed=False, tests_run=0)
    if len(valid) != len(tests):
        return dict(status="unsupported_test_types", passed=False, tests_run=0)
    env = dict(os.environ, MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen", DISPLAY="", WAYLAND_DISPLAY="", BROWSER="true")
    with tempfile.TemporaryDirectory(prefix="lcb-lite-") as td:
        path = Path(td) / "solution.py"
        path.write_text(code)
        for i, case in enumerate(valid):
            try:
                result = subprocess.run([sys.executable, str(path)], cwd=td, env=env,
                                        input=case.get("input", ""), capture_output=True,
                                        text=True, timeout=timeout)
                got, want = result.stdout.strip(), str(case.get("output", "")).strip()
                if result.returncode or got != want:
                    return dict(status="fail", passed=False, tests_run=i + 1, available_tests=len(valid),
                                failures=[dict(i=i, returncode=result.returncode, got=got[:500],
                                               want=want[:500], stderr=result.stderr[-500:])])
            except subprocess.TimeoutExpired:
                return dict(status="fail", passed=False, tests_run=i + 1, available_tests=len(valid),
                            failures=[dict(i=i, timeout=True)])
    return dict(status="pass", passed=True, tests_run=len(valid), available_tests=len(valid), failures=[])
