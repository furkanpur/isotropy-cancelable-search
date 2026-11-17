"""Download the pretrained backbone (facebook/dinov2-base) into <out>/dinov2-base-<rev12>/ and record FETCH.json.

Runs in a container with network access and NO data mounted.  FETCH.json holds the resolved revision, the repository id
and the SHA-256 of every file; the weights directory is then mounted read-only by train/embed (offline).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = "facebook/dinov2-base"
PATTERNS = ["config.json", "model.safetensors", "preprocessor_config.json"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="directory that receives dinov2-base-<rev12>/")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download
    with tempfile.TemporaryDirectory() as cache:
        snap = Path(snapshot_download(a.repo, cache_dir=cache, allow_patterns=PATTERNS))
        revision = snap.name
        dest = Path(a.out) / f"dinov2-base-{revision[:12]}"
        if dest.exists():
            print(f"refusing to overwrite {dest}", file=sys.stderr)
            return 1
        dest.mkdir(parents=True)
        for f in sorted(snap.iterdir()):
            shutil.copyfile(f.resolve(), dest / f.name)
    files = {p.name: _sha256(p) for p in sorted(dest.iterdir())}
    (dest / "FETCH.json").write_text(json.dumps({"repo": a.repo, "revision": revision, "files_sha256": files},
                                                indent=2, sort_keys=True) + "\n")
    print(f"fetched {a.repo}@{revision} into {dest}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
