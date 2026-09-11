from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime import DimerSwinDetector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the pinned DIMER Swin-T Mask R-CNN detector.")
    parser.add_argument("images", nargs="+", help="Image path(s) to score")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--provenance", help="Optional provenance JSON path")
    parser.add_argument("--cache-dir", default=".dimer-models")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int, default=300)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detector = DimerSwinDetector(cache_dir=args.cache_dir, device=args.device)
    rows = [
        d.to_dict()
        for d in detector.predict_many(
            args.images,
            score_threshold=args.score_threshold,
            max_detections=args.max_detections,
        )
    ]
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, indent=2) + "\n")
    if args.provenance:
        detector.write_provenance(args.provenance)
    print(json.dumps({"images": len(args.images), "detections": len(rows), "output": str(target)}))


if __name__ == "__main__":
    main()
