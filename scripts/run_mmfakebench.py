#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cage.experiments.runner import MMFakeBenchRunConfig, MMFakeBenchRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal CAGE/T2Agent experiment on MMFakeBench.")
    parser.add_argument("--root", default="/data/lhy_data/MMFakeBench", help="MMFakeBench root directory.")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split.")
    parser.add_argument("--output", default=None, help="Prediction JSONL output path.")
    parser.add_argument("--metrics-output", default=None, help="Metrics JSON output path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in the split.")
    parser.add_argument(
        "--sample-mode",
        default="sequential",
        choices=["sequential", "balanced"],
        help="Sampling mode. balanced selects equal true/fake examples when --max-samples is set.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for balanced sampling.")
    parser.add_argument("--no-resume", action="store_true", help="Disable JSONL resume and overwrite output.")
    parser.add_argument("--load-image", action="store_true", help="Decode images in the dataset wrapper.")
    parser.add_argument(
        "--pass-image-to-agent",
        action="store_true",
        help="Pass sample.image_path into T2Agent. For zipped MMFakeBench images this is usually false unless paths are extracted.",
    )
    parser.add_argument(
        "--prediction-strategy",
        default="fake_cls_baseline",
        choices=["fake_cls_baseline", "agent"],
        help="Prediction strategy. fake_cls_baseline is a sanity baseline: original->true, otherwise->fake.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or f"outputs/mmfakebench_{args.split}_predictions.jsonl"
    config = MMFakeBenchRunConfig(
        root=args.root,
        split=args.split,
        output_path=output,
        metrics_path=args.metrics_output,
        max_samples=args.max_samples,
        start_index=args.start_index,
        sample_mode=args.sample_mode,
        seed=args.seed,
        resume=not args.no_resume,
        load_image=args.load_image,
        pass_image_to_agent=args.pass_image_to_agent,
        prediction_strategy=args.prediction_strategy,
    )
    runner = MMFakeBenchRunner(config=config)
    metrics = runner.run()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
