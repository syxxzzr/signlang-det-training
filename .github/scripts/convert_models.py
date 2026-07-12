#!/usr/bin/env python3
"""Convert the exported sign-language encoder from PyTorch to ONNX and RKNN."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from rknn.api import RKNN
except ImportError:  # Unit tests and PT/ONNX-only development do not require RKNN.
    RKNN = None


CHECKPOINT_SCHEMA = 1
PREPROCESSING_CONTRACT = "hand168-temporal"
FEATURE_SHAPE = (1, 64, 168)
LENGTH_SHAPE = (1,)
OUTPUT_SHAPE = (1, 64, 128)


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation, groups=channels
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.pointwise(self.depthwise(inputs)))) + inputs


class HandEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 168,
        hand_dim: int = 96,
        fusion_dim: int = 192,
        embedding_dim: int = 128,
        max_frames: int = 64,
    ):
        super().__init__()
        if feature_dim != 168:
            raise ValueError("hand168-temporal requires feature_dim=168")
        self.model_config = {
            "feature_dim": feature_dim,
            "hand_dim": hand_dim,
            "fusion_dim": fusion_dim,
            "embedding_dim": embedding_dim,
            "max_frames": max_frames,
        }
        self.input_projection = nn.Linear(84, hand_dim)
        self.tcn = nn.Sequential(
            TemporalBlock(hand_dim, 1),
            TemporalBlock(hand_dim, 2),
            TemporalBlock(hand_dim, 4),
        )
        self.fusion = nn.Linear(hand_dim * 2, fusion_dim)
        self.position = nn.Parameter(torch.zeros(1, max_frames, fusion_dim))
        layer = nn.TransformerEncoderLayer(
            fusion_dim,
            4,
            fusion_dim * 4,
            0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.output = nn.Linear(fusion_dim, embedding_dim)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = features.shape[1]
        mask = torch.arange(steps, device=features.device)[None] >= lengths[:, None]
        inputs = features.masked_fill(mask[..., None], 0.0).reshape(
            features.shape[0], steps, 2, 84
        )
        streams = []
        for hand in range(2):
            stream = self.input_projection(inputs[:, :, hand]).transpose(1, 2)
            streams.append(self.tcn(stream).transpose(1, 2))
        fused = self.fusion(torch.cat(streams, dim=-1)) + self.position[:, :steps]
        fused = self.transformer(fused, src_key_padding_mask=mask)
        output = F.normalize(self.output(fused), dim=-1)
        return output.masked_fill(mask[..., None], 0.0)


def encoder_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_encoder(path: Path) -> HandEncoder:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "schema",
        "preprocessing",
        "model_config",
        "encoder",
        "encoder_fingerprint",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"PT export is missing fields: {missing}")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported PT schema: {payload['schema']}")
    if payload["preprocessing"] != PREPROCESSING_CONTRACT:
        raise ValueError(f"Unsupported preprocessing contract: {payload['preprocessing']}")
    actual = encoder_fingerprint(payload["encoder"])
    if actual != payload["encoder_fingerprint"]:
        raise ValueError("PT encoder fingerprint does not match its weights")
    model = HandEncoder(**payload["model_config"])
    model.load_state_dict(payload["encoder"], strict=True)
    return model.eval()


def export_onnx(pt_path: Path, onnx_path: Path) -> None:
    model = load_encoder(pt_path)
    generator = torch.Generator().manual_seed(20260712)
    features = torch.randn(FEATURE_SHAPE, generator=generator, dtype=torch.float32)
    lengths = torch.tensor([47], dtype=torch.int32)
    features[:, 47:] = -100.0
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(features, lengths).cpu().numpy()
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        try:
            # The fused eval fast-path emits aten::_transformer_encoder_layer_fwd,
            # which the opset 17 exporter cannot lower. The unfused path is
            # numerically equivalent and consists of standard ONNX operations.
            torch.backends.mha.set_fastpath_enabled(False)
            torch.onnx.export(
                model,
                (features, lengths),
                onnx_path,
                input_names=["features", "lengths"],
                output_names=["frame_embeddings"],
                opset_version=17,
                do_constant_folding=True,
            )
        finally:
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    import onnx
    import onnxruntime as ort

    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["frame_embeddings"],
        {"features": features.numpy(), "lengths": lengths.numpy()},
    )[0]
    if tuple(actual.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"ONNX output shape is {tuple(actual.shape)}, expected {OUTPUT_SHAPE}")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


def _safe_members(bundle: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = bundle.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise ValueError(f"Calibration archive contains an unsafe member: {member.name}")
    return members


def prepare_calibration_dataset(archive: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, members=_safe_members(bundle))

    source_list = destination / "dataset.txt"
    if not source_list.is_file():
        raise ValueError("Calibration archive does not contain dataset.txt")
    resolved_lines = []
    for line_number, line in enumerate(source_list.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if len(values) != 2:
            raise ValueError(f"Calibration dataset line {line_number} must contain two inputs")
        paths = [(destination / value).resolve() for value in values]
        if any(destination.resolve() not in path.parents for path in paths):
            raise ValueError(f"Calibration dataset line {line_number} contains an unsafe path")
        if not all(path.is_file() for path in paths):
            raise ValueError(f"Calibration dataset line {line_number} references a missing input")
        features = np.load(paths[0], allow_pickle=False)
        lengths = np.load(paths[1], allow_pickle=False)
        if features.dtype != np.float32 or tuple(features.shape) != FEATURE_SHAPE:
            raise ValueError(f"Calibration features on line {line_number} have an invalid contract")
        if lengths.dtype != np.int32 or tuple(lengths.shape) != LENGTH_SHAPE:
            raise ValueError(f"Calibration lengths on line {line_number} have an invalid contract")
        if not 1 <= int(lengths[0]) <= FEATURE_SHAPE[1]:
            raise ValueError(f"Calibration length on line {line_number} is out of range")
        resolved_lines.append(" ".join(str(path) for path in paths))
    if not resolved_lines:
        raise ValueError("Calibration dataset is empty")
    resolved = destination / "dataset.resolved.txt"
    resolved.write_text("\n".join(resolved_lines) + "\n", encoding="utf-8")
    return resolved


def _require_success(result: object, operation: str) -> None:
    if result not in (None, 0):
        raise RuntimeError(f"RKNN {operation} failed with code {result}")


def _build_one_rknn(
    onnx_path: Path,
    output_path: Path,
    target_platform: str,
    dataset: Optional[Path],
) -> None:
    if RKNN is None:
        raise RuntimeError("rknn-toolkit2 is required for RKNN conversion")
    converter = RKNN(verbose=False)
    try:
        _require_success(converter.config(target_platform=target_platform), "config")
        _require_success(converter.load_onnx(model=str(onnx_path)), "load_onnx")
        build_args = {"do_quantization": dataset is not None}
        if dataset is not None:
            build_args["dataset"] = str(dataset)
        _require_success(converter.build(**build_args), "build")
        _require_success(converter.export_rknn(str(output_path)), "export_rknn")
    finally:
        converter.release()


def build_rknn_models(
    onnx_path: Path,
    dataset: Path,
    rknn_path: Path,
    int8_rknn_path: Path,
    target_platform: str,
) -> None:
    _build_one_rknn(onnx_path, rknn_path, target_platform, None)
    _build_one_rknn(onnx_path, int8_rknn_path, target_platform, dataset)


def convert(
    pt_path: Path,
    calibration_archive: Path,
    output_dir: Path,
    target_platform: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "signlang_det_encoder.onnx"
    rknn_path = output_dir / "signlang_det_encoder.rknn"
    int8_path = output_dir / "signlang_det_encoder.int8.rknn"
    export_onnx(pt_path, onnx_path)
    with tempfile.TemporaryDirectory() as directory:
        dataset = prepare_calibration_dataset(calibration_archive, Path(directory) / "calibration")
        build_rknn_models(onnx_path, dataset, rknn_path, int8_path, target_platform)
    return [onnx_path, rknn_path, int8_path]


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--pt", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--target-platform", default="rk3588")
    return command


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(argv)
    convert(args.pt, args.calibration, args.output_dir, args.target_platform)


if __name__ == "__main__":
    main()
