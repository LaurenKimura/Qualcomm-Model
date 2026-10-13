"""Write the 1-2 page report the judges asked for, in plain language."""

from __future__ import annotations

from pathlib import Path


def write_report(path: Path, team: str, baseline: dict, winner: dict, candidates: list[dict], notes: list[str], class_names: list[str]):
    discarded = _discarded_lines(baseline, winner, candidates)
    worst_name = _class_name(class_names, winner.get("worst_class_index", 0))
    kept_pct = 100 * winner["accuracy"] / max(baseline["accuracy"], 1e-9)
    kept_label = "practice only" if baseline["accuracy"] < 0.02 else f"{kept_pct:.1f}% kept"
    lines = [
        f"# {team}: shrinking a satellite-photo classifier",
        "",
        "The starting model already names the land in a satellite photo. "
        "Our job was to keep that skill and cut the size, the wait, and the energy.",
        "",
    ]
    if any("not the contest" in note for note in notes):
        lines.extend([
            "This file is a practice run. The official contest checkpoint was not in the folder, so the last answer layer was a random guess. A score near zero here is expected. Do not hand this folder in.",
            "",
        ])
    lines.extend([
        "## What we kept",
        "",
        winner["why"],
        "",
        "## Scoreboard numbers",
        "",
        "| | Starting model | Submitted model | How many times smaller or faster |",
        "|---|---:|---:|---:|",
        _row("Accuracy (share of photos named correctly)", f"{baseline['accuracy']*100:.2f}%", f"{winner['accuracy']*100:.2f}%", kept_label),
        _row("Worst land type", f"{baseline['worst_class_accuracy']*100:.2f}%", f"{winner['worst_class_accuracy']*100:.2f}% ({worst_name})", ""),
        _row("Parameter count (millions)", f"{baseline['params_million']:.2f}", f"{winner['params_million']:.2f}", f"{winner['params_ratio']:.2f}x fewer"),
        _row("File size (MB)", f"{baseline['size_mb']:.2f}", f"{winner['size_mb']:.2f}", f"{winner['size_ratio']:.2f}x smaller"),
        _row("Arithmetic per photo (billion MACs)", f"{baseline['macs_giga']:.3f}", f"{winner['macs_giga']:.3f}", f"{baseline['macs_giga']/max(winner['macs_giga'],1e-9):.2f}x less"),
        _row("Wait per photo (ms)", f"{baseline['latency_ms']:.2f}", f"{winner['latency_ms']:.2f}", f"{baseline['latency_ms']/max(winner['latency_ms'],1e-9):.2f}x faster"),
        _row("Peak memory (MB)", f"{baseline['peak_ram_mb']:.1f}", f"{winner['peak_ram_mb']:.1f}", f"{baseline['peak_ram_mb']/max(winner['peak_ram_mb'],1e-9):.2f}x"),
        _row("Energy per photo (joules)", f"{baseline['joules_per_inference']:.4f}", f"{winner['joules_per_inference']:.4f}", f"{baseline['joules_per_inference']/max(winner['joules_per_inference'],1e-12):.2f}x less"),
        "",
        _energy_note(winner.get("codecarbon_mode", "tdp_estimate")),
        "",
        "## What we tried and did not submit",
        "",
        *discarded,
        "",
        "## Decisions made before any of those runs",
        "",
        "- Unstructured pruning (writing zeros but keeping the slots) was not used. The file size and the wait do not change.",
        "- Dynamic quantization was not used. The contest notes say Qualcomm AI Hub often rejects those files, and they can be slower than the original on a CPU.",
        "- The input the judges feed us stays 224 by 224. When a smaller working size won, the shrink happens inside the file.",
        "- The heavy teacher network is already baked into the starting checkpoint. A long second distillation does not fit a CPU-only day. Recovery training is a short patch after pruning, and it uses the teacher only if you pass --use-teacher and the teacher file is present.",
        "",
        "## How a judge reruns this",
        "",
        "1. Install Python 3.11 and run `pip install -r requirements.txt` with the CPU PyTorch index from the contest packet.",
        "2. Place `student_mobilenetv2_distilled.pt`, `val_split.json`, and the EuroSAT `data` folder next to `run_pipeline.py`. The teacher file is optional.",
        "3. Run `python run_pipeline.py --team-name \"Your Team\"`.",
        "4. The script writes `submission/student_model_int8.onnx`, `submission/metrics.json`, this report, and `submission/tradeoff.csv`.",
        "",
        "## Notes from this run",
        "",
        *(f"- {note}" for note in notes),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _energy_note(mode: str) -> str:
    if mode == "measured":
        return "Energy meter mode: measured. The laptop read a real power sensor. Baseline and final were measured on the same machine."
    if mode == "not measured":
        return "Energy was not measured on this run. On contest day, leave off --skip-energy and do not pass --quick."
    return "Energy meter mode: tdp_estimate. The laptop could not read a power sensor, so joules are wait-time times a typical wattage. The contest allows that. Baseline and final were measured on the same machine with the same method."


def _row(name, before, after, ratio) -> str:
    return f"| {name} | {before} | {after} | {ratio} |"


def _class_name(class_names: list[str], index: int) -> str:
    if 0 <= index < len(class_names):
        return class_names[index]
    return f"class {index}"


def _discarded_lines(baseline: dict, winner: dict, candidates: list[dict]) -> list[str]:
    lines = []
    others = [item for item in candidates if item.get("name") != winner.get("name") and "accuracy" in item]
    if not others:
        return ["Every other setting failed to run. The errors are in the notes."]
    # Show the most aggressive shrink and the closest runner-up.
    by_shrink = sorted(others, key=lambda item: item.get("params_million", 1e9))
    by_score = sorted(others, key=lambda item: item.get("score", 0), reverse=True)
    shown = []
    for item in (by_score[0], by_shrink[0]):
        if item["name"] not in {entry["name"] for entry in shown}:
            shown.append(item)
    for item in shown:
        acc_drop = (baseline["accuracy"] - item["accuracy"]) * 100
        if acc_drop >= 0:
            change = f"accuracy fell by {acc_drop:.2f} percentage points"
        else:
            change = f"accuracy rose by {-acc_drop:.2f} percentage points"
        lines.append(
            f"- {item['plain']}: {change} "
            f"({item['accuracy']*100:.2f}% versus the start at {baseline['accuracy']*100:.2f}%). "
            f"Parameters became {item.get('params_ratio', 0):.2f}x smaller and the file {item.get('size_ratio', 0):.2f}x smaller. "
            + item.get("reject_reason", "We did not submit it because another setting scored higher on the combined tradeoff.")
        )
    lines.append("The full curve is in `tradeoff.csv` next to this report. One row was written after every setting.")
    return lines
