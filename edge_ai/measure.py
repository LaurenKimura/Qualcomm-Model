"""Count how good, how big, how slow, and how hungry a model is.

The contest uses these numbers as the "before" and "after" on the score sheet.
Baseline and final model must be measured the same way on the same computer.
"""

from __future__ import annotations

import io
import logging
import os
import platform
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from thop import profile


def count_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_megabytes(model) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 * 1024)


def file_megabytes(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def macs_and_params(model, spatial_size: int = 224):
    """MACs are multiply-adds: how much arithmetic one photo costs."""
    clone = _clone_for_profile(model)
    dummy = torch.randn(1, 3, spatial_size, spatial_size)
    macs, params = profile(clone, inputs=(dummy,), verbose=False)
    return float(macs), float(params)


def _clone_for_profile(model):
    # thop attaches temporary hooks. Profiling a copy leaves the real model alone.
    import copy

    clone = copy.deepcopy(model)
    clone.eval()
    return clone


def accuracy_and_confusion(predict_batch, loader, num_classes: int = 10, max_batches: int | None = None):
    """predict_batch(images) -> a LongTensor of class ids, one per photo."""
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    seen = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        predictions = predict_batch(images)
        for true_label, predicted in zip(labels, predictions):
            confusion[int(true_label), int(predicted)] += 1
            seen += 1
    if seen == 0:
        raise RuntimeError("The data loader produced no photos.")
    correct = int(confusion.diag().sum().item())
    per_class = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1)
    worst = float(per_class.min().item())
    worst_index = int(per_class.argmin().item())
    return {
        "accuracy": correct / seen,
        "worst_class_accuracy": worst,
        "worst_class_index": worst_index,
        "per_class_accuracy": [float(value) for value in per_class],
        "images": seen,
    }


def pytorch_predictor(model):
    model.eval()

    def predict(images):
        with torch.no_grad():
            return model(images).argmax(dim=1).cpu()

    return predict


def onnx_predictor(session):
    """The saved file is built for one photo at a time, which is what the judges send."""
    input_name = session.get_inputs()[0].name

    def predict(images):
        array = images.numpy()
        if array.shape[0] == 1:
            logits = session.run(None, {input_name: array})[0]
            return torch.from_numpy(np.argmax(logits, axis=1))
        chosen = []
        for row in array:
            logits = session.run(None, {input_name: row[None, ...]})[0]
            chosen.append(int(np.argmax(logits)))
        return torch.tensor(chosen, dtype=torch.long)

    return predict


def latency_ms(run_once, repeats: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        run_once()
    start = time.perf_counter()
    for _ in range(repeats):
        run_once()
    return (time.perf_counter() - start) / repeats * 1000


def peak_rss_mb(run_once, repeats: int = 50) -> float:
    """Sample memory while the model runs.

    Demo: one reading before and one after misses the high point in the
    middle. A background sampler records the maximum while photos are scored.
    """
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss / (1024 * 1024)
    sampling = True

    def sample():
        nonlocal peak
        while sampling:
            peak = max(peak, process.memory_info().rss / (1024 * 1024))
            time.sleep(0.01)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        for _ in range(repeats):
            run_once()
    finally:
        sampling = False
        thread.join()
    return peak


def energy_per_inference(run_once, repeats: int = 100) -> tuple[float, str]:
    """Return joules for one photo, and whether the meter was real or a guess.

    On many Windows laptops CodeCarbon cannot see the power sensor. It then
    multiplies time by a typical chip wattage. The contest calls that a
    TDP estimate and does not penalize it, but we have to say which one we got.
    """
    from codecarbon import EmissionsTracker

    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("codecarbon")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    tracker = _make_tracker()
    mode = "tdp_estimate"
    try:
        tracker.start()
        for _ in range(repeats):
            run_once()
        tracker.stop()
    finally:
        logger.removeHandler(handler)

    text = log_buffer.getvalue().lower()
    # RAPL and powermetrics read real sensors. Anything else is the fallback.
    if any(word in text for word in ("rapl", "powermetrics", "nvml", "intel power gadget")):
        mode = "measured"
    if any(word in text for word in ("tdp", "fallback", "default power")):
        mode = "tdp_estimate"

    joules = (tracker.final_emissions_data.energy_consumed * 3.6e6) / repeats
    return float(joules), mode


def _make_tracker():
    from codecarbon import EmissionsTracker
    import inspect

    preferred = {
        "measure_power_secs": 1,
        "log_level": "error",
        "save_to_file": False,
        "country_iso_code": "USA",
        "allow_multiple_runs": True,
        "project_name": "minimalist-edge-ai",
    }
    accepted = inspect.signature(EmissionsTracker.__init__).parameters
    kwargs = {key: value for key, value in preferred.items() if key in accepted}
    return EmissionsTracker(**kwargs)


def machine_description() -> str:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu = platform.processor() or platform.machine()
    return f"{platform.system()} {platform.release()}, {cpu}, {ram_gb:.0f} GB RAM, CPU only"


def dummy_input():
    return torch.randn(1, 3, 224, 224)
