"""Run the contest pipeline from the provided checkpoint to a submission folder."""

from __future__ import annotations

import csv
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from edge_ai.compress import (
    ResolutionStudent,
    collect_calibration_images,
    export_onnx,
    load_student,
    load_teacher,
    quantize_static,
    recover,
    structured_prune,
)
from edge_ai.measure import (
    accuracy_and_confusion,
    count_params,
    energy_per_inference,
    file_megabytes,
    latency_ms,
    machine_description,
    macs_and_params,
    onnx_predictor,
    peak_rss_mb,
    pytorch_predictor,
    state_dict_megabytes,
)
from edge_ai.report import write_report
from edge_ai.score import pick_winner, score_candidate


def run(args):
    root = Path(args.root).resolve()
    if args.self_test:
        self_test(root)
        return

    from edge_ai.data import load_eurosat, make_loaders

    print("Step 1 of 6. Loading the satellite photos and the starting network.")
    dataset = load_eurosat(root / args.data, download=args.download)
    train_loader, val_loader, official_split = make_loaders(
        dataset, root / args.split, args.batch_size
    )
    class_names = list(getattr(dataset, "classes", []))
    student, weight_source = load_student(_optional(root / args.student), demo_imagenet=args.demo_imagenet)
    teacher = load_teacher(_optional(root / args.teacher)) if args.use_teacher else None
    notes = [
        f"Weights: {weight_source}.",
        "Validation split: official val_split.json." if official_split else "Validation split: temporary 80/20 split with seed 42, because val_split.json was not in the folder. Judges will not accept numbers from this split.",
    ]
    if teacher is None:
        notes.append("Teacher network was not used. The starting model is already distilled. Pass --use-teacher only if teacher_resnet50_eurosat.pt is available and you can spend the extra CPU time.")

    out = root / "outputs"
    submission = root / "submission"
    out.mkdir(parents=True, exist_ok=True)
    submission.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.finetune_steps = 0
        args.skip_energy = True

    print("Step 2 of 6. Measuring the starting network, before any shrinking.")
    print("Accuracy, speed, and energy for the start and the finish both come from ONNX Runtime, so the comparison is fair.")
    baseline_model = ResolutionStudent(student, 224).eval()
    macs, params = macs_and_params(baseline_model)
    baseline = {
        "name": "baseline",
        "plain": "starting model, full 224 photo, no channel cut, 32-bit file",
        "params_million": params / 1e6,
        "macs_giga": macs / 1e9,
        "latency_ms": 0.0,
        "peak_ram_mb": 0.0,
        "joules_per_inference": 0.0,
        "codecarbon_mode": "tdp_estimate",
    }
    baseline_onnx = out / "baseline_fp32.onnx"
    export_onnx(baseline_model, baseline_onnx)
    baseline_limit = args.probe_batches if args.quick else None
    baseline.update(_onnx_accuracy(baseline_onnx, val_loader, baseline_limit))
    baseline.update(_measure_onnx_runtime(baseline_onnx, repeats=20 if args.quick else 50, skip_energy=args.skip_energy))
    baseline["size_mb"] = file_megabytes(baseline_onnx)
    if (root / args.student).is_file():
        baseline["checkpoint_mb"] = file_megabytes(root / args.student)
    print(_plain_metrics("Starting point", baseline))

    print("Step 3 of 6. Trying smaller photos and thinner networks. Accuracy is checked on a sample so this finishes on a laptop.")
    pruned = {}
    for ratio in args.prune_ratios:
        try:
            print(f"  Cutting {ratio:.0%} of channels...")
            pruned[ratio] = structured_prune(student, ratio) if ratio > 0 else student
            after = count_params(pruned[ratio])
            print(f"  Parameters now {after/1e6:.2f} million (started at {count_params(student)/1e6:.2f} million).")
        except Exception as error:
            notes.append(f"Pruning at {ratio:.0%} failed and was discarded: {error}")
            print(f"  Could not cut {ratio:.0%} of channels. Saved the error and kept going.")

    probes = []
    for ratio, model in pruned.items():
        for size in args.sizes:
            name = f"size{size}_prune{ratio:.2f}"
            wrapped = ResolutionStudent(model, size).eval()
            try:
                measured = _measure_pytorch(
                    name,
                    wrapped,
                    val_loader,
                    max_batches=args.probe_batches,
                    measure_runtime=False,
                    skip_energy=True,
                )
            except Exception as error:
                notes.append(f"{name} failed during the accuracy check: {error}")
                continue
            measured["name"] = name
            measured["image_size"] = size
            measured["prune_ratio"] = ratio
            measured["quantized"] = False
            measured["plain"] = _plain_name(size, ratio, quantized=False)
            # Runtime is filled in for real finalists. Copy the baseline so an
            # unfinished row cannot pretend to be faster.
            for key in ("latency_ms", "peak_ram_mb", "joules_per_inference", "codecarbon_mode"):
                measured[key] = baseline[key]
            measured["size_mb"] = state_dict_megabytes(model)
            score_candidate(measured, baseline)
            probes.append(measured)
            print(
                f"  {measured['plain']}: {measured['accuracy']*100:.1f}% correct, "
                f"{measured['params_ratio']:.2f}x fewer parameters."
            )

    if not probes:
        raise RuntimeError("Every shrink setting failed. See outputs notes.")

    print("Step 4 of 6. Rebuilding the best few as real ONNX files, including 8-bit versions.")
    finalists = _choose_finalists(probes, baseline, limit=1 if args.quick else 3)
    calibration = collect_calibration_images(train_loader, 4 if args.quick else args.calib_images)
    finished = []
    for probe in finalists:
        finished.extend(_build_finalist(probe, pruned, train_loader, teacher, args, out, calibration, val_loader, baseline, notes))

    if not finished:
        raise RuntimeError("Could not export any finalist. See the notes in the report.")

    print("Step 5 of 6. Picking the submitted model.")
    for item in finished:
        score_candidate(item, baseline)
    winner = pick_winner(finished, baseline)
    winner["why"] = _why(winner, baseline)
    if winner.get("accuracy_kept", 0) < 0.97:
        notes.append(
            "No candidate kept 97% of the starting accuracy. "
            "The submitted file is the best combined score anyway, and the curve shows where accuracy fell off."
        )

    print("Step 6 of 6. Writing the submission folder.")
    submitted_path = submission / ("student_model_int8.onnx" if winner["quantized"] else "student_model_fp32.onnx")
    submitted_path.write_bytes(Path(winner["onnx_path"]).read_bytes())
    winner["size_mb"] = file_megabytes(submitted_path)
    score_candidate(winner, baseline)

    metrics = _metrics_json(args.team_name, baseline, winner)
    (submission / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_curve(submission / "tradeoff.csv", [baseline, *probes, *finished])
    _plot_curve(submission / "tradeoff.png", probes + finished, winner)
    write_report(submission / "REPORT.md", args.team_name, baseline, winner, probes + finished, notes, class_names)
    (out / "candidates.json").write_text(json.dumps(probes + finished, indent=2), encoding="utf-8")
    print()
    print(_plain_metrics("Submitted", winner))
    print(f"Folder to hand in: {submission}")


def self_test(root: Path):
    """Prove the shrinking steps work, without the contest photos or checkpoint."""
    from torchvision import models
    from torch import nn

    print("Plumbing test. This does not use the contest checkpoint or the satellite photos.")
    print("It checks that thinning the network and saving an 8-bit file actually get smaller.")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.eval()
    before_params = count_params(model)

    print("Cutting 30% of channels...")
    pruned = structured_prune(model, 0.30)
    after_params = count_params(pruned)
    if after_params >= before_params:
        raise RuntimeError("Pruning did not reduce the parameter count.")

    wrapped = ResolutionStudent(pruned, 128).eval()
    with torch.no_grad():
        output = wrapped(torch.randn(2, 3, 224, 224))
    if tuple(output.shape) != (2, 10):
        raise RuntimeError(f"Expected answers for 2 photos, got {tuple(output.shape)}.")

    out = root / "outputs" / "self_test"
    out.mkdir(parents=True, exist_ok=True)
    fp32_path = export_onnx(wrapped, out / "practice_fp32.onnx")
    images = np.random.randn(4, 3, 224, 224).astype(np.float32)
    int8_path = quantize_static(fp32_path, out / "practice_int8.onnx", images)

    import onnxruntime as ort

    session = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"input": images[:1]})[0]
    if result.shape != (1, 10):
        raise RuntimeError(f"8-bit file returned shape {result.shape}.")

    fp32_mb = file_megabytes(fp32_path)
    int8_mb = file_megabytes(int8_path)
    print()
    print(f"Parameters: {before_params/1e6:.2f} million -> {after_params/1e6:.2f} million ({before_params/after_params:.2f}x fewer).")
    print(f"File size:  {fp32_mb:.2f} MB -> {int8_mb:.2f} MB ({fp32_mb/int8_mb:.2f}x smaller).")
    print("The 8-bit file accepted a 224 by 224 photo and returned 10 scores.")
    print("Plumbing test passed. Contest numbers still need the official checkpoint and val_split.json.")
    summary = {
        "params_before": before_params,
        "params_after": after_params,
        "fp32_mb": fp32_mb,
        "int8_mb": int8_mb,
        "int8_output_shape": list(result.shape),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_finalist(probe, pruned, train_loader, teacher, args, out, calibration, val_loader, baseline, notes):
    ratio = probe["prune_ratio"]
    size = probe["image_size"]
    base_model = pruned[ratio]
    results = []
    model = ResolutionStudent(_copy_module(base_model), size)
    if args.finetune_steps > 0:
        print(f"  Short recovery training for { _plain_name(size, ratio, False) }...")
        try:
            model = recover(model, train_loader, steps=args.finetune_steps, teacher=teacher)
        except Exception as error:
            notes.append(f"Recovery training failed for prune {ratio:.0%} size {size}: {error}")
            print(f"  Recovery training failed: {error}")

    stem = out / f"size{size}_prune{ratio:.2f}"
    try:
        fp32_path = export_onnx(model.eval(), stem.with_suffix(".onnx"))
        fp32_item = _candidate_from_onnx(
            name=f"size{size}_prune{ratio:.2f}_fp32",
            plain=_plain_name(size, ratio, quantized=False),
            model=model,
            onnx_path=fp32_path,
            image_size=size,
            prune_ratio=ratio,
            quantized=False,
            loader=val_loader,
            max_batches=None if not args.quick else args.probe_batches,
            skip_energy=args.skip_energy,
            quick=args.quick,
        )
        score_candidate(fp32_item, baseline)
        results.append(fp32_item)
    except Exception as error:
        notes.append(f"Float export failed for size {size} prune {ratio:.0%}: {error}")
        print(f"  Float export failed: {error}")
        return results

    try:
        int8_path = quantize_static(fp32_path, stem.with_name(stem.name + "_int8.onnx"), calibration)
        int8_item = _candidate_from_onnx(
            name=f"size{size}_prune{ratio:.2f}_int8",
            plain=_plain_name(size, ratio, quantized=True),
            model=model,
            onnx_path=int8_path,
            image_size=size,
            prune_ratio=ratio,
            quantized=True,
            loader=val_loader,
            max_batches=None if not args.quick else args.probe_batches,
            skip_energy=args.skip_energy,
            quick=args.quick,
        )
        score_candidate(int8_item, baseline)
        results.append(int8_item)
        print(f"  8-bit file: {int8_item['accuracy']*100:.1f}% correct, {int8_item['size_mb']:.2f} MB.")
    except Exception as error:
        notes.append(f"8-bit conversion failed for size {size} prune {ratio:.0%}: {error}\n{traceback.format_exc()}")
        print(f"  8-bit conversion failed: {error}")
    return results


def _candidate_from_onnx(name, plain, model, onnx_path, image_size, prune_ratio, quantized, loader, max_batches, skip_energy, quick):
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    stats = accuracy_and_confusion(onnx_predictor(session), loader, max_batches=max_batches)
    macs, _params = macs_and_params(model)
    runtime = _measure_onnx_runtime(onnx_path, repeats=10 if quick else 50, skip_energy=skip_energy)

    def once():
        session.run(None, {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)})

    # Latency above used random input. Accuracy used real photos. Both are the ONNX file.
    del once
    item = {
        "name": name,
        "plain": plain,
        "image_size": image_size,
        "prune_ratio": prune_ratio,
        "quantized": quantized,
        "onnx_path": str(onnx_path),
        "params_million": count_params(model) / 1e6,
        "size_mb": file_megabytes(onnx_path),
        "macs_giga": macs / 1e9,
        "reject_reason": "Another setting had a better combined score.",
    }
    item.update(stats)
    item.update(runtime)
    return item


def _measure_pytorch(name, model, loader, max_batches, measure_runtime: bool, skip_energy: bool):
    stats = accuracy_and_confusion(pytorch_predictor(model), loader, max_batches=max_batches)
    macs, params = macs_and_params(model)
    item = {
        "name": name,
        "plain": name,
        "accuracy": stats["accuracy"],
        "worst_class_accuracy": stats["worst_class_accuracy"],
        "worst_class_index": stats["worst_class_index"],
        "per_class_accuracy": stats["per_class_accuracy"],
        "params_million": params / 1e6,
        "macs_giga": macs / 1e9,
        "size_mb": state_dict_megabytes(model),
        "latency_ms": 0.0,
        "peak_ram_mb": 0.0,
        "joules_per_inference": 0.0,
        "codecarbon_mode": "tdp_estimate",
    }
    if measure_runtime:
        dummy = torch.randn(1, 3, 224, 224)

        def run_once():
            with torch.no_grad():
                model(dummy)

        item["latency_ms"] = latency_ms(run_once, repeats=20 if max_batches else 50)
        item["peak_ram_mb"] = peak_rss_mb(run_once, repeats=20 if max_batches else 30)
        if not skip_energy:
            joules, mode = energy_per_inference(run_once, repeats=30 if max_batches else 100)
            item["joules_per_inference"] = joules
            item["codecarbon_mode"] = mode
    return item


def _onnx_accuracy(path: Path, loader, max_batches):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return accuracy_and_confusion(onnx_predictor(session), loader, max_batches=max_batches)


def _measure_onnx_runtime(path: Path, repeats: int, skip_energy: bool):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

    def run_once():
        session.run(None, {"input": dummy})

    item = {
        "latency_ms": latency_ms(run_once, repeats=repeats),
        "peak_ram_mb": peak_rss_mb(run_once, repeats=min(20, repeats)),
        "joules_per_inference": 0.0,
        "codecarbon_mode": "not measured",
    }
    if not skip_energy:
        joules, mode = energy_per_inference(run_once, repeats=max(30, repeats))
        item["joules_per_inference"] = joules
        item["codecarbon_mode"] = mode
    return item


def _choose_finalists(probes, baseline, limit: int):
    def rank(item):
        kept = item["accuracy"] / max(baseline["accuracy"], 1e-9)
        shrink = item["params_ratio"] * max(baseline["macs_giga"], 1e-9) / max(item["macs_giga"], 1e-9)
        # Settings that hold accuracy sort above settings that do not.
        return (1 if kept >= 0.97 else 0, shrink if kept >= 0.97 else kept)

    ordered = sorted(probes, key=rank, reverse=True)
    chosen = []
    for item in ordered:
        if len(chosen) >= limit:
            break
        chosen.append(item)
    # Always also rebuild the most compressed setting that still got 90% of
    # baseline accuracy, so the report can show the cliff instead of hiding it.
    aggressive = sorted(probes, key=lambda item: item["params_million"])
    for item in aggressive:
        kept = item["accuracy"] / max(baseline["accuracy"], 1e-9)
        if kept >= 0.90 and item["name"] not in {entry["name"] for entry in chosen}:
            chosen.append(item)
            break
    return chosen


def _metrics_json(team: str, baseline: dict, winner: dict):
    def pack(item):
        return {
            "accuracy": round(item["accuracy"], 6),
            "params_million": round(item["params_million"], 4),
            "size_mb": round(item["size_mb"], 4),
            "macs_giga": round(item["macs_giga"], 4),
            "latency_ms": round(item["latency_ms"], 4),
            "peak_ram_mb": round(item["peak_ram_mb"], 4),
            "joules_per_inference": round(item["joules_per_inference"], 6),
        }

    techniques = []
    if winner["image_size"] != 224:
        techniques.append(f"internal resize to {winner['image_size']}")
    if winner["prune_ratio"] > 0:
        techniques.append("structured channel pruning")
    if winner["quantized"]:
        techniques.append("static QDQ int8 quantization")
    if not techniques:
        techniques.append("none")
    return {
        "team_name": team,
        "baseline": pack(baseline),
        "final": pack(winner),
        "worst_class_accuracy": round(winner["worst_class_accuracy"], 6),
        "codecarbon_mode": winner.get("codecarbon_mode", "tdp_estimate"),
        "techniques_applied": techniques,
        "ai_hub_profile_summary": "Not run yet. Run profile_hub.py after qai-hub configure. If the service is down, submit these local numbers and say so.",
        "measurement_machine": machine_description(),
    }


def _why(winner, baseline) -> str:
    pieces = []
    if winner["image_size"] != 224:
        pieces.append(
            f"The contest photos start as 64 by 64 pixels and are stretched to 224. "
            f"We accept the required 224 by 224 input, then shrink it to {winner['image_size']} inside the file."
        )
    else:
        pieces.append("We kept the working photo at 224 by 224 because smaller sizes lost too much accuracy.")
    if winner["prune_ratio"] > 0:
        pieces.append(
            f"We deleted about {winner['prune_ratio']:.0%} of the channels, the ones with the smallest weights, and left the first layer and the 10-answer layer intact."
        )
    else:
        pieces.append("Channel deletion cost more accuracy than it saved, so the submitted network keeps the original channels.")
    if winner["quantized"]:
        pieces.append("Weights are stored as 8-bit integers in the QDQ format AI Hub expects, calibrated on training photos.")
    else:
        pieces.append("This file stays 32-bit. The 8-bit copy is on the curve and scored worse on the combined tradeoff, usually because it was slower or less accurate.")
    if baseline["accuracy"] < 0.02:
        pieces.append(
            f"The starting network could not name the photos, because the contest checkpoint is not in the folder yet. "
            f"The file is still {winner['size_ratio']:.2f}x smaller."
        )
    else:
        pieces.append(
            f"It keeps {winner['accuracy_kept']*100:.1f}% of the starting accuracy "
            f"and is {winner['size_ratio']:.2f}x smaller on disk."
        )
    return " ".join(pieces)


def _plain_name(size: int, ratio: float, quantized: bool) -> str:
    shrink = "no channel cut" if ratio <= 0 else f"{ratio:.0%} of channels removed"
    bits = "8-bit file" if quantized else "32-bit file"
    photo = "full 224 photo" if size == 224 else f"photo shrunk to {size} inside the model"
    return f"{photo}, {shrink}, {bits}"


def _plain_metrics(title, item) -> str:
    return (
        f"{title}: {item['accuracy']*100:.2f}% correct, "
        f"worst land type {item['worst_class_accuracy']*100:.2f}%, "
        f"{item['params_million']:.2f}M parameters, {item['size_mb']:.2f} MB, "
        f"{item['latency_ms']:.1f} ms."
    )


def _write_curve(path: Path, rows: list[dict]):
    fields = [
        "name",
        "plain",
        "accuracy",
        "worst_class_accuracy",
        "params_million",
        "size_mb",
        "macs_giga",
        "latency_ms",
        "peak_ram_mb",
        "joules_per_inference",
        "score",
        "params_ratio",
        "size_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _plot_curve(path: Path, rows: list[dict], winner: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = [row for row in rows if row.get("size_ratio")]
    if not usable:
        return
    figure, axis = plt.subplots(figsize=(7, 4.5))
    xs = [row["size_ratio"] for row in usable]
    ys = [row["accuracy"] * 100 for row in usable]
    axis.scatter(xs, ys, color="#3d5a40")
    axis.scatter([winner["size_ratio"]], [winner["accuracy"] * 100], color="#c44900", s=80, zorder=3)
    axis.set_xlabel("How many times smaller the file is")
    axis.set_ylabel("Photos named correctly (%)")
    axis.set_title("Tradeoff curve (orange point is the one we submit)")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _copy_module(module):
    import copy

    return copy.deepcopy(module)


def _optional(path: Path):
    return path if path.is_file() else None
