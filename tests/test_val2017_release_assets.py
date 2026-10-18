"""Every pinned release-asset URL resolves, and every image parses in its declared format (no model is run).

Network test: it downloads the 50 sample images (~8.3 MB) from this repository's GitHub release through the
notebook's own sample cell, which checks each file's size and SHA-256, then decodes every file with Pillow and
compares format and size with the pin. Set DIMER_OFFLINE=1 to skip it on a machine without network access.
"""
# ruff: noqa: E501  -- long literal digests and test names are kept on one line
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "swin_detection_task_inference.ipynb"
PIN = json.loads((ROOT / "tools" / "coco_val2017_sample_50.json").read_text(encoding="utf-8"))
RELEASE = f"https://github.com/kurtvalcorza/swin-detection-pipeline/releases/download/{PIN['release_tag']}/"

pytestmark = pytest.mark.skipif(os.environ.get("DIMER_OFFLINE") == "1", reason="DIMER_OFFLINE=1")


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "dimer-ci-asset-check"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def test_release_metadata_assets_resolve_and_match_the_pin() -> None:
    manifest = json.loads(_get(RELEASE + "coco_val2017_sample_50.json"))
    assert manifest == PIN
    attribution = _get(RELEASE + "ATTRIBUTION.md").decode("utf-8")
    for image in PIN["images"]:
        assert image["file_name"] in attribution and image["flickr_url"] in attribution


def test_every_pinned_image_resolves_and_parses_through_the_notebook_cell(tmp_path: Path, monkeypatch) -> None:
    import numpy
    from PIL import Image

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "SAMPLE = 'val2017'  # @param" in "".join(c["source"]))
    monkeypatch.chdir(tmp_path)
    namespace = {"numpy": numpy, "__name__": "notebook_cell"}
    exec(source, namespace)  # noqa: S102 - the notebook's own sample cell, default path
    assert namespace["sample_ready"] is True and namespace["sample_kind"] == "COCO-val2017-50"
    assert len(namespace["ground_truth"]) == 50 and namespace["n_ground_truth_boxes"] == 279
    assert namespace["OUTPUT_SUFFIX"] == ""
    for image, path in zip(PIN["images"], namespace["image_paths"], strict=True):
        assert path.name == image["file_name"] and path.stat().st_size == image["bytes"]
        with Image.open(path) as opened:
            opened.load()
            assert opened.format == image["format"] and opened.size == (image["width"], image["height"])
