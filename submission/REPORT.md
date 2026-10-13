# REPLACE_ME: shrinking a satellite-photo classifier

The starting model already names the land in a satellite photo. Our job was to keep that skill and cut the size, the wait, and the energy.

## What we kept

The contest photos start as 64 by 64 pixels and are stretched to 224. We accept the required 224 by 224 input, then shrink it to 160 inside the file. Channel deletion cost more accuracy than it saved, so the submitted network keeps the original channels. This file stays 32-bit. The 8-bit copy is on the curve and scored worse on the combined tradeoff, usually because it was slower or less accurate. It keeps 99.4% of the starting accuracy and is 1.00x smaller on disk.

## Scoreboard numbers

| | Starting model | Submitted model | How many times smaller or faster |
|---|---:|---:|---:|
| Accuracy (share of photos named correctly) | 99.19% | 98.63% | 99.4% kept |
| Worst land type | 95.85% | 96.05% (PermanentCrop) |  |
| Parameter count (millions) | 2.24 | 2.24 | 1.00x fewer |
| File size (MB) | 8.51 | 8.51 | 1.00x smaller |
| Arithmetic per photo (billion MACs) | 0.326 | 0.166 | 1.96x less |
| Wait per photo (ms) | 4.96 | 2.57 | 1.93x faster |
| Peak memory (MB) | 470.7 | 478.1 | 0.98x |
| Energy per photo (joules) | 0.1585 | 0.0959 | 1.65x less |

Energy meter mode: tdp_estimate. The laptop could not read a power sensor, so joules are wait-time times a typical wattage. The contest allows that. Baseline and final were measured on the same machine with the same method.

## What we tried and did not submit

- photo shrunk to 160 inside the model, no channel cut, 8-bit file: accuracy fell by 4.78 percentage points (94.41% versus the start at 99.19%). Parameters became 1.00x smaller and the file 3.30x smaller. Another setting had a better combined score.
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
- Refine path: 160-pixel working size, class-weighted repair for River, Pasture, and the crop classes, and static 8-bit convolutions with the final layer left at full precision.
