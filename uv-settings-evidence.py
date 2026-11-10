"""Future admitted settings-only real uv children; authored NOT RUN.

Exercises the captured bootstrap/worker/uv environment boundaries, not a PM
installation or proof of canonical setup subprocesses. No full env/output saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    args = parser.parse_args()
    # Future gate must set cwd/PYTHONPATH to the new frozen task tree.
    from pm.environment import PythonEnvironment, _base_environment
    from pm.runtime import runtime_environment

    import pm.environment
    assert Path(pm.environment.__file__).resolve().is_relative_to(args.tree.resolve())
    caps = {key: "1" for key in ("UV_CONCURRENT_DOWNLOADS", "UV_CONCURRENT_BUILDS",
                               "UV_CONCURRENT_INSTALLS", "UV_CONCURRENT_CACHE_READS")}
    assert all(os.environ.get(key) == value for key, value in caps.items())
    args.private.mkdir(mode=0o700, parents=True, exist_ok=False)
    original_run = subprocess.run
    receipts = []
    for stage, source, no_config in (("bootstrap", args.tree / "pm", True),
                                      ("app", args.tree, False)):
        env = _base_environment(runtime_environment())
        environment = PythonEnvironment(uv=args.uv, python=args.python,
                                        destination=args.private / (stage + "-venv"),
                                        cache=args.private / "cache", env=env, no_config=no_config)

        def observe(command, **kwargs):
            actual = kwargs["env"]
            assert {key: actual[key] for key in caps} == caps
            with subprocess.Popen(command, cwd=kwargs["cwd"], env=actual,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding="utf-8", errors="replace") as child:
                identity = original_run(["ps", "-p", str(child.pid), "-o", "pid=,ppid=,lstart=,pgid="],
                                        capture_output=True, text=True, check=True).stdout.strip()
                try:
                    stdout, stderr = child.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    raise
                # Show-settings exits before installation. Retain only allowlisted settings.
                text = stdout + stderr
                resolved = {}
                for field in ("downloads", "builds", "installs", "cache_reads"):
                    values = re.findall(r"\b" + field + r":\s*(\d+)", text)
                    assert values and set(values) == {"1"}, field
                    resolved[field] = 1
                assert child.returncode == 0, "settings-only uv child failed"
                receipts.append({"stage": stage, "binary": str(args.uv.resolve()),
                                 "binary_sha256": hashlib.sha256(args.uv.read_bytes()).hexdigest(),
                                 "pid_identity": identity, "no_config": no_config,
                                 "forwarded_values": caps, "resolved_settings": resolved,
                                 "exit": child.returncode})
                return subprocess.CompletedProcess(command, child.returncode, "", "")

        # Observe the real _run boundary; observer launches the actual pinned uv binary.
        with patch("pm.environment.subprocess.run", side_effect=observe):
            environment._run(["sync", "--frozen", "--show-settings", "--python", str(args.python)],
                             cwd=source, timeout=30)
    (args.private / "settings-only.json").write_text(json.dumps(receipts, indent=2) + "\n")


if __name__ == "__main__":
    main()
