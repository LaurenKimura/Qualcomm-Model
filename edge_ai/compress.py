"""Shrink the network in ways the judges can actually see.

Three tools, used in this order:

1. Shrink the photo inside the model. The contest photos are 64 by 64 pixels
   that were stretched to 224. Looking at the stretched copy costs a lot of
   math and adds almost no new detail. The submitted file still accepts a
   224 by 224 photo, which is required.

2. Structured pruning. Delete whole channels the network barely uses, then
   rebuild the smaller layers. Writing zeros in place does not shrink the file.
   That is why this uses torch-pruning instead of torch.nn.utils.prune.

3. Static int8 quantization. Store each weight in 8 bits instead of 32.
   The file gets about 4 times smaller. Calibration uses real training photos
   so the smaller numbers still point at the right answer. The export is QDQ,
   which is the form Qualcomm AI Hub accepts. Dynamic quantization is avoided
   on purpose: the contest notes say device compilers often reject it.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F


# Demo: these land types look alike after the photo is shrunk.
# AnnualCrop, Highway, Pasture, PermanentCrop, and River get a heavier penalty
# during the short repair lesson so the weak class is not ignored.
WEAK_CLASS_WEIGHTS = [1.4, 1.0, 1.0, 1.3, 1.0, 2.0, 1.8, 1.0, 2.2, 1.0]


class ResolutionStudent(nn.Module):
    """Accepts the required 224 by 224 photo, then shrinks it before the real work.

    Demo: the contest photos are 64 by 64 stretched to 224. A river is a thin
    line. 128 pixels blurs that line into a road or a lake. 160 is the smallest
    size that kept every land type above the starting worst class.
    """

    def __init__(self, student: nn.Module, image_size: int):
        super().__init__()
        self.student = student
        self.image_size = int(image_size)

    def forward(self, x):
        # Demo: judges still send 224 by 224. The shrink happens inside the file.
        if self.image_size != 224:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.student(x)


def load_student(checkpoint: Path | None, demo_imagenet: bool):
    """Build the 10-class MobileNetV2 the contest starts from."""
    from torchvision import models

    if checkpoint is not None and checkpoint.is_file():
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 10)
        model.load_state_dict(_load_state(checkpoint))
        model.eval()
        return model, "official checkpoint"

    if not demo_imagenet:
        raise FileNotFoundError(
            "student_mobilenetv2_distilled.pt was not found. "
            "On contest day, put that file next to run_pipeline.py. "
            "To practice before the file exists, pass --demo-imagenet."
        )

    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    # The downloaded network names 1000 everyday objects. Replace only the
    # last layer so it has 10 answers, one per land type. Those 10 answers
    # start random. This is a practice brain, not the contest brain.
    model.classifier[1] = nn.Linear(model.last_channel, 10)
    model.eval()
    return model, "practice ImageNet weights (not the contest checkpoint)"


def load_teacher(checkpoint: Path | None):
    if checkpoint is None or not checkpoint.is_file():
        return None
    from torchvision import models

    teacher = models.resnet50(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    teacher.load_state_dict(_load_state(checkpoint))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _load_state(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def structured_prune(student: nn.Module, ratio: float) -> nn.Module:
    """Delete `ratio` of the channels and return a physically smaller network.

    ratio=0.30 means "drop about 30% of the channels." The final 10-way
    answer layer is never pruned. The first convolution is left alone too:
    it is small, and damaging it blurs every later answer.
    """
    if ratio <= 0:
        return copy.deepcopy(student).eval()

    import torch_pruning as tp

    model = copy.deepcopy(student).eval()
    example = torch.randn(1, 3, 224, 224)
    pruner = _make_pruner(tp, model, example, ratio, _ignored_layers(model))
    pruner.step()

    with torch.no_grad():
        output = model(example)
    if tuple(output.shape) != (1, 10):
        raise RuntimeError(f"Pruned model returned shape {tuple(output.shape)}, expected (1, 10).")
    model.eval()
    return model


def _ignored_layers(model: nn.Module):
    ignored = []
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == 10:
            ignored.append(module)
        if first_conv is None and isinstance(module, nn.Conv2d):
            first_conv = module
    if first_conv is not None:
        ignored.append(first_conv)
    return ignored


def _make_pruner(tp, model, example, ratio, ignored):
    importance_class = getattr(tp.importance, "GroupMagnitudeImportance", None)
    if importance_class is None:
        importance_class = tp.importance.MagnitudeImportance
    importance = importance_class(p=2)

    pruner_class = getattr(tp.pruner, "BasePruner", None) or tp.pruner.MagnitudePruner
    accepted = inspect.signature(pruner_class.__init__).parameters
    kwargs = {}
    if "importance" in accepted:
        kwargs["importance"] = importance
    if "pruning_ratio" in accepted:
        kwargs["pruning_ratio"] = ratio
    elif "ch_sparsity" in accepted:
        kwargs["ch_sparsity"] = ratio
    if "ignored_layers" in accepted:
        kwargs["ignored_layers"] = ignored
    if "global_pruning" in accepted:
        kwargs["global_pruning"] = True
    if "round_to" in accepted:
        kwargs["round_to"] = 8
    if "iterative_steps" in accepted:
        kwargs["iterative_steps"] = 1
    if "max_pruning_ratio" in accepted:
        kwargs["max_pruning_ratio"] = 0.8
    return pruner_class(model, example, **kwargs)


def recover(model, train_loader, steps: int, teacher=None, lr: float = 1e-4, class_weights=None):
    """A short study session after a change.

    Demo: eighty updates is a patch, not a new training run. A full lesson
    will not finish on a CPU in one day. The class weights make a miss on
    River or Pasture count more than a miss on Forest.
    """
    if steps <= 0:
        model.eval()
        return model

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    weight = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
    hard_loss = nn.CrossEntropyLoss(weight=weight)
    soft_loss = nn.KLDivLoss(reduction="batchmean")
    temperature = 4.0
    done = 0
    while done < steps:
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = hard_loss(logits, labels)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                loss = 0.5 * loss + 0.5 * (
                    soft_loss(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(teacher_logits / temperature, dim=1),
                    )
                    * (temperature * temperature)
                )
            loss.backward()
            optimizer.step()
            done += 1
            if done >= steps:
                break
    model.eval()
    return model


def export_onnx(model: nn.Module, path: Path) -> Path:
    """Save the network as an ONNX file. Judges run this file, not PyTorch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    try:
        torch.onnx.export(model, dummy, str(path), **export_kwargs)
    except TypeError:
        export_kwargs.pop("dynamo")
        torch.onnx.export(model, dummy, str(path), **export_kwargs)
    _strip_duplicate_value_info(path)
    return path


def _strip_duplicate_value_info(path: Path):
    """AI Hub rejects graphs whose internal names collide with the output name."""
    model = onnx.load(str(path))
    output_names = {item.name for item in model.graph.output}
    kept = [item for item in model.graph.value_info if item.name not in output_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    onnx.save(model, str(path))


class ListCalibrationReader:
    """Hands ONNX Runtime a few real photos so it can pick int8 scale factors."""

    def __init__(self, images: np.ndarray, input_name: str = "input"):
        if images.ndim == 3:
            images = images[None, ...]
        self.images = images.astype(np.float32)
        self.input_name = input_name
        self.index = 0

    def get_next(self):
        if self.index >= len(self.images):
            return None
        batch = self.images[self.index : self.index + 1]
        self.index += 1
        return {self.input_name: batch}

    def rewind(self):
        self.index = 0


def collect_calibration_images(loader, limit: int) -> np.ndarray:
    chosen = []
    for images, _labels in loader:
        for image in images:
            chosen.append(image.numpy())
            if len(chosen) >= limit:
                return np.stack(chosen)
    if not chosen:
        raise RuntimeError("No calibration photos were available.")
    return np.stack(chosen)


def quantize_static(onnx_path: Path, output_path: Path, images: np.ndarray) -> Path:
    """Turn a float ONNX file into a static QDQ int8 file.

    Demo: only the convolutions are stored as 8-bit integers. The last layer,
    the one that prints the 10 land-type scores, stays at full precision.
    Quantizing that layer was what knocked Pasture and River down. The scales
    are chosen with entropy on real training photos, which is steadier than
    taking the single darkest and brightest pixel.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static as ort_quantize

    prepared = output_path.with_name(output_path.stem + "_prepared.onnx")
    _prepare_for_quant(onnx_path, prepared)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shared = dict(
        model_input=str(prepared),
        model_output=str(output_path),
        calibration_data_reader=ListCalibrationReader(images),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        # Conv weights are almost the whole file. Gemm is the 10-answer layer.
        op_types_to_quantize=["Conv"],
    )
    try:
        shared["calibration_data_reader"].rewind()
        ort_quantize(calibrate_method=CalibrationMethod.Entropy, **shared)
    except Exception:
        shared["calibration_data_reader"].rewind()
        ort_quantize(calibrate_method=CalibrationMethod.MinMax, **shared)
    _strip_duplicate_value_info(output_path)
    return output_path


def _prepare_for_quant(source: Path, dest: Path):
    try:
        from onnxruntime.quantization.preprocess import quant_pre_process
    except ImportError:
        from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(source), str(dest), skip_optimization=False)
    except TypeError:
        quant_pre_process(str(source), str(dest))
    except Exception:
        model = onnx.load(str(source))
        inferred = onnx.shape_inference.infer_shapes(model)
        onnx.save(inferred, str(dest))
