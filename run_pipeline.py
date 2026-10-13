"""One command for the Minimalist Edge AI Challenge.

Demo path, in this order:
1. The starting file is a MobileNetV2 that already names 10 kinds of land.
2. We still accept a 224 by 224 photo, then shrink it to 160 inside the file.
   128 was faster on paper and made River the worst class.
3. A short repair lesson pays extra attention to River, Pasture, and crops.
4. The convolutions are saved as 8-bit integers. The 10-answer layer stays
   at full precision, which is what kept the weak classes from collapsing.
5. Channel deletion is measured and shown. Heavy cuts make a tiny file and
   forget the quiz, so they are not the submitted file.

Contest morning:

    .\\edge-ai-env\\Scripts\\python.exe run_pipeline.py --team-name "Your Team"

Apply the fixes above to the checkpoint already in this folder:

    .\\edge-ai-env\\Scripts\\python.exe run_pipeline.py --refine --team-name "Your Team"
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
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 160, 128])
    parser.add_argument("--prune-ratios", type=float, nargs="+", default=[0.0, 0.10, 0.25])
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--demo-imagenet", action="store_true", help="Practice with public ImageNet weights if the contest checkpoint is missing.")
    parser.add_argument("--use-teacher", action="store_true", help="Also learn from the ResNet-50 teacher during the short recovery. Slower.")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Tiny practice run. Do not submit these numbers.")
    parser.add_argument("--refine", action="store_true", help="Rebuild the submission at 160 pixels with a class-weighted repair and conv-only 8-bit weights.")
    parser.add_argument("--self-test", action="store_true", help="Check pruning and 8-bit export with a random network. No photos required.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
