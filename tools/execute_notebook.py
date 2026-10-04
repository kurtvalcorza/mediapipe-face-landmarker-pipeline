#!/usr/bin/env python3
"""Execute a copy of the tutorial notebook with form fields set (NOTEBOOK_SPEC 2.2 §5.1, EXE1-EXE7).

The executor never edits the committed notebook. It copies it, rewrites the named ``# @param`` form fields in the copy,
fails if a field is not found exactly once (so a renamed field breaks the job instead of silently running the default),
and runs the copy top to bottom with ``jupyter nbconvert --execute`` in a working directory of your choice.

Usage:
    python tools/execute_notebook.py --workdir /tmp/run                       # the default path, fields untouched
    python tools/execute_notebook.py --workdir /tmp/run --set USE_BYOD=True --set "BYOD_PATH='/data/face.jpg'"

Requires ``jupyter nbconvert`` and an ``ipykernel`` in the interpreter that runs this script (or ``--jupyter``).
"""
# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "mediapipe_face_landmarker_colab.ipynb"


def set_fields(notebook: dict, fields: dict[str, str]) -> dict:
    for name, value in fields.items():
        hits = 0
        pattern = re.compile(rf"^{re.escape(name)} = .*?(  # @param.*)$", re.M)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            new, n = pattern.subn(lambda m, k=name, v=value: f"{k} = {v}{m.group(1)}", source)
            if n:
                hits += n
                cell["source"] = new
        if hits != 1:
            raise SystemExit(f"form field {name!r} found {hits} times; expected exactly once (EXE6)")
    return notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="a Python literal for one form field")
    parser.add_argument("--output", default="executed.ipynb")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--jupyter", default=None, help="path to the jupyter executable (default: <python> -m jupyter)")
    args = parser.parse_args(argv)
    fields = dict(item.split("=", 1) for item in args.set)
    notebook = set_fields(json.loads(args.notebook.read_text(encoding="utf-8")), fields)
    args.workdir.mkdir(parents=True, exist_ok=True)
    copy = args.workdir / "run_copy.ipynb"
    copy.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    command = [args.jupyter] if args.jupyter else [sys.executable, "-m", "jupyter"]
    command += ["nbconvert", "--to", "notebook", "--execute", f"--ExecutePreprocessor.timeout={args.timeout}", "--output", args.output, str(copy)]
    print({"executing": str(copy), "fields": fields}, flush=True)
    return subprocess.run(command, cwd=args.workdir, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
