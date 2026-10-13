"""One command for the Minimalist Edge AI Challenge.

Contest morning, after the starter files are in this folder:

    .\\edge-ai-env\\Scripts\\python.exe run_pipeline.py --team-name "Your Team"

Before the starter files exist, check that shrinking works:

    .\\edge-ai-env\\Scripts\\python.exe run_pipeline.py --self-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_ai.pipeline import run


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Shrink the contest MobileNetV2 and write a submission folder.")
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--team-name", default="REPLACE_ME")
    parser.add_argument("--student", default="student_mobilenetv2_distilled.pt")
    parser.add_argument("--teacher", default="teacher_resnet50_eurosat.pt")
    parser.add_argument("--split", default="val_split.json")
    parser.add_argument("--data", default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-batches", type=int, default=8, help="Photo batches used to rank settings. The finalists are then scored on the full validation set.")
    parser.add_argument("--calib-images", type=int, default=64, help="Training photos used to calibrate 8-bit numbers.")
    parser.add_argument("--finetune-steps", type=int, default=80, help="Recovery updates after pruning. Keep this small on a CPU.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 160, 128, 96])
    parser.add_argument("--prune-ratios", type=float, nargs="+", default=[0.0, 0.25, 0.45, 0.65])
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--demo-imagenet", action="store_true", help="Practice with public ImageNet weights if the contest checkpoint is missing.")
    parser.add_argument("--use-teacher", action="store_true", help="Also learn from the ResNet-50 teacher during the short recovery. Slower.")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Tiny practice run. Do not submit these numbers.")
    parser.add_argument("--self-test", action="store_true", help="Check pruning and 8-bit export with a random network. No photos required.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
