# REPLACE_ME: shrinking a satellite-photo classifier

The starting model already names the land in a satellite photo. Our job was to keep that skill and cut the size, the wait, and the energy.

## What we kept

The contest photos start as 64 by 64 pixels and are stretched to 224. We accept the required 224 by 224 input, then shrink it to 128 inside the file. Channel deletion cost more accuracy than it saved, so the submitted network keeps the original channels. This file stays 32-bit. The 8-bit copy is on the curve and scored worse on the combined tradeoff, usually because it was slower or less accurate. It keeps 98.8% of the starting accuracy and is 1.00x smaller on disk.

## Scoreboard numbers

| | Starting model | Submitted model | How many times smaller or faster |
|---|---:|---:|---:|
| Accuracy (share of photos named correctly) | 99.19% | 98.04% | 98.8% kept |
| Worst land type | 95.85% | 95.17% (River) |  |
| Parameter count (millions) | 2.24 | 2.24 | 1.00x fewer |
| File size (MB) | 8.51 | 8.51 | 1.00x smaller |
| Arithmetic per photo (billion MACs) | 0.326 | 0.107 | 3.06x less |
| Wait per photo (ms) | 4.47 | 3.17 | 1.41x faster |
| Peak memory (MB) | 391.5 | 631.9 | 0.62x |
| Energy per photo (joules) | 0.1778 | 0.0853 | 2.08x less |

Energy meter mode: tdp_estimate. The laptop could not read a power sensor, so joules are wait-time times a typical wattage. The contest allows that. Baseline and final were measured on the same machine with the same method.

## What we tried and did not submit

- photo shrunk to 160 inside the model, no channel cut, 32-bit file: accuracy fell by 0.41 percentage points (98.78% versus the start at 99.19%). Parameters became 1.00x smaller and the file 1.00x smaller. Another setting had a better combined score.
- full 224 photo, 65% of channels removed, 32-bit file: accuracy fell by 92.54 percentage points (6.64% versus the start at 99.19%). Parameters became 10.86x smaller and the file 9.17x smaller. We did not submit it because another setting scored higher on the combined tradeoff.
The full curve is in `tradeoff.csv` next to this report. One row was written after every setting.

## Decisions made before any of those runs

- Unstructured pruning (writing zeros but keeping the slots) was not used. The file size and the wait do not change.
- Dynamic quantization was not used. The contest notes say Qualcomm AI Hub often rejects those files, and they can be slower than the original on a CPU.
- The input the judges feed us stays 224 by 224. When a smaller working size won, the shrink happens inside the file.
- The heavy teacher network is already baked into the starting checkpoint. A long second distillation does not fit a CPU-only day. Recovery training is a short patch after pruning, and it uses the teacher only if you pass --use-teacher and the teacher file is present.

## How a judge reruns this

1. Install Python 3.11 and run `pip install -r requirements.txt` with the CPU PyTorch index from the contest packet.
2. Place `student_mobilenetv2_distilled.pt`, `val_split.json`, and the EuroSAT `data` folder next to `run_pipeline.py`. The teacher file is optional.
3. Run `python run_pipeline.py --team-name "Your Team"`.
4. The script writes `submission/student_model_int8.onnx`, `submission/metrics.json`, this report, and `submission/tradeoff.csv`.

## Notes from this run

- Weights: official checkpoint.
- Validation split: temporary 80/20 split with seed 42, because val_split.json was not in the folder. Judges will not accept numbers from this split.
- Teacher network was not used. The starting model is already distilled. Pass --use-teacher only if teacher_resnet50_eurosat.pt is available and you can spend the extra CPU time.
