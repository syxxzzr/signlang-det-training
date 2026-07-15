#!/usr/bin/env python3
"""Convert the exported sign-language encoder from PyTorch to ONNX and RKNN."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
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
OUTPUT_SHAPE = (1, 64, 128)
PADDING_VALUE = -100.0
PADDING_DETECTION_THRESHOLD = -50.0


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

    def forward_with_padding_mask(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        steps = features.shape[1]
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

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = features.shape[1]
        mask = torch.arange(steps, device=features.device)[None] >= lengths[:, None]
        return self.forward_with_padding_mask(features, mask)


class DeploymentEncoder(nn.Module):
    """Expose the runtime contract while keeping lengths internal to the export graph."""

    def __init__(self, encoder: HandEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        # A midpoint threshold preserves the external padding contract after
        # conversion while conforming valid features remain far above it.
        padding_mask = torch.all(landmarks < PADDING_DETECTION_THRESHOLD, dim=-1)
        lengths = torch.sum(~padding_mask, dim=1).to(torch.int32)
        return self.encoder(landmarks, lengths)


def infer_lengths_from_landmarks(landmarks: np.ndarray) -> np.ndarray:
    values = np.asarray(landmarks)
    if values.dtype != np.float32 or tuple(values.shape) != FEATURE_SHAPE:
        raise ValueError(
            f"Landmarks must be float32 with shape [1, 64, 168], received "
            f"dtype={values.dtype}, shape={values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Landmarks contain non-finite values")
    padding_mask = np.all(values == PADDING_VALUE, axis=-1)
    lengths = np.sum(~padding_mask, axis=1, dtype=np.int32)
    if np.any((lengths < 1) | (lengths > FEATURE_SHAPE[1])):
        raise ValueError(f"Landmark valid lengths are out of range: {lengths.tolist()}")
    expected_mask = np.arange(FEATURE_SHAPE[1])[None] >= lengths[:, None]
    if not np.array_equal(padding_mask, expected_mask):
        raise ValueError("Landmark padding must be contiguous full -100.0 frames on the right")
    return lengths.astype(np.int32, copy=False)


def encoder_fingerprint(state_dict: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_safe_checkpoint(path: Path) -> Mapping[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("PT export payload must be a mapping")
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
    encoder = payload["encoder"]
    if not isinstance(encoder, Mapping) or not encoder:
        raise ValueError("PT encoder state must be a non-empty mapping")
    if not all(isinstance(name, str) and isinstance(tensor, torch.Tensor) for name, tensor in encoder.items()):
        raise ValueError("PT encoder state must contain only named tensors")
    actual = encoder_fingerprint(encoder)
    if actual != payload["encoder_fingerprint"]:
        raise ValueError("PT encoder fingerprint does not match its weights")
    return payload


def load_encoder(path: Path) -> HandEncoder:
    payload = load_safe_checkpoint(path)
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported PT schema: {payload['schema']}")
    if payload["preprocessing"] != PREPROCESSING_CONTRACT:
        raise ValueError(f"Unsupported preprocessing contract: {payload['preprocessing']}")
    model = HandEncoder(**payload["model_config"])
    model.load_state_dict(payload["encoder"], strict=True)
    return model.eval()


def export_onnx(
    pt_path: Path, onnx_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    encoder = load_encoder(pt_path)
    model = DeploymentEncoder(encoder).eval()
    generator = torch.Generator().manual_seed(20260712)
    landmarks = torch.randn(FEATURE_SHAPE, generator=generator, dtype=torch.float32)
    lengths = torch.tensor([47], dtype=torch.int32)
    landmarks[:, 47:] = PADDING_VALUE
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = model(landmarks).cpu().numpy()
        internal = encoder(landmarks, lengths).cpu().numpy()
        np.testing.assert_allclose(expected, internal, rtol=0.0, atol=0.0)
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        try:
            # The fused eval fast-path emits aten::_transformer_encoder_layer_fwd,
            # which the opset 17 exporter cannot lower. The unfused path is
            # numerically equivalent and consists of standard ONNX operations.
            torch.backends.mha.set_fastpath_enabled(False)
            torch.onnx.export(
                model,
                (landmarks,),
                onnx_path,
                input_names=["landmarks"],
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
    graph_inputs = [value.name for value in graph.graph.input]
    graph_outputs = [value.name for value in graph.graph.output]
    if graph_inputs != ["landmarks"] or graph_outputs != ["frame_embeddings"]:
        raise RuntimeError(
            f"ONNX deployment contract mismatch: inputs={graph_inputs}, outputs={graph_outputs}"
        )
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    runtime_inputs = session.get_inputs()
    runtime_outputs = session.get_outputs()
    if (
        len(runtime_inputs) != 1
        or runtime_inputs[0].name != "landmarks"
        or runtime_inputs[0].type != "tensor(float)"
        or runtime_inputs[0].shape != list(FEATURE_SHAPE)
        or len(runtime_outputs) != 1
        or runtime_outputs[0].name != "frame_embeddings"
        or runtime_outputs[0].type != "tensor(float)"
        or runtime_outputs[0].shape != list(OUTPUT_SHAPE)
    ):
        raise RuntimeError(
            "ONNX Runtime deployment metadata does not match the single-input contract"
        )
    actual = session.run(["frame_embeddings"], {"landmarks": landmarks.numpy()})[0]
    if tuple(actual.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"ONNX output shape is {tuple(actual.shape)}, expected {OUTPUT_SHAPE}")
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    return landmarks.numpy(), actual


def _require_success(result: object, operation: str) -> None:
    if result not in (None, 0):
        raise RuntimeError(f"RKNN {operation} failed with code {result}")


def _validate_rknn_outputs(
    model_name: str,
    outputs: object,
    test_inputs: Sequence[np.ndarray],
    expected: np.ndarray,
) -> None:
    if len(test_inputs) != 1:
        raise RuntimeError(
            f"RKNN deployment validation received {len(test_inputs)} inputs, expected 1"
        )
    if not isinstance(outputs, (list, tuple)):
        raise RuntimeError(
            f"RKNN inference returned {type(outputs).__name__}, expected a list of outputs"
        )
    if len(outputs) != 1:
        output_count = len(outputs)
        raise RuntimeError(f"RKNN inference returned {output_count} outputs, expected 1")
    actual = np.asarray(outputs[0])
    if tuple(actual.shape) != OUTPUT_SHAPE:
        raise RuntimeError(f"RKNN output shape is {tuple(actual.shape)}, expected {OUTPUT_SHAPE}")
    if not np.isfinite(actual).all():
        raise RuntimeError("RKNN inference produced non-finite values")
    valid_steps = int(infer_lengths_from_landmarks(test_inputs[0])[0])
    expected_valid = expected[:, :valid_steps].astype(np.float64, copy=False).reshape(-1)
    actual_valid = actual[:, :valid_steps].astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(expected_valid) * np.linalg.norm(actual_valid)
    cosine = float(np.dot(expected_valid, actual_valid) / denominator) if denominator else 0.0
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=5e-4)
    if valid_steps < OUTPUT_SHAPE[1]:
        np.testing.assert_allclose(
            actual[:, valid_steps:], 0.0, rtol=0.0, atol=1e-6
        )
    print(
        f"Validated {model_name} with RKNN simulator: "
        f"dtype={actual.dtype}, cosine={cosine:.6f}"
    )


def _build_rknn(
    onnx_path: Path,
    output_path: Path,
    target_platform: str,
    test_inputs: Sequence[np.ndarray],
    expected: np.ndarray,
) -> None:
    if RKNN is None:
        raise RuntimeError("rknn-toolkit2 is required for RKNN conversion")
    converter = RKNN(verbose=False)
    try:
        _require_success(converter.config(target_platform=target_platform), "config")
        _require_success(converter.load_onnx(model=str(onnx_path)), "load_onnx")
        _require_success(converter.build(do_quantization=False), "build")
        _require_success(converter.export_rknn(str(output_path)), "export_rknn")
        _require_success(converter.init_runtime(), "init_runtime")
        outputs = converter.inference(inputs=list(test_inputs))
        _validate_rknn_outputs(output_path.name, outputs, test_inputs, expected)
    finally:
        converter.release()


def build_rknn_model(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str,
    test_inputs: Sequence[np.ndarray],
    expected: np.ndarray,
) -> None:
    _build_rknn(onnx_path, rknn_path, target_platform, test_inputs, expected)


def convert(
    pt_path: Path,
    output_dir: Path,
    target_platform: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "signlang_det_encoder.onnx"
    rknn_path = output_dir / "signlang_det_encoder.rknn"
    landmarks, expected = export_onnx(pt_path, onnx_path)
    build_rknn_model(
        onnx_path, rknn_path, target_platform, [landmarks], expected
    )
    return [onnx_path, rknn_path]


def validate_inputs(pt_path: Path) -> None:
    load_safe_checkpoint(pt_path)


def onnx_reference(onnx_path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    import onnxruntime as ort

    generator = torch.Generator().manual_seed(20260712)
    landmarks = torch.randn(FEATURE_SHAPE, generator=generator, dtype=torch.float32).numpy()
    landmarks[:, 47:] = PADDING_VALUE
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    runtime_inputs = session.get_inputs()
    if len(runtime_inputs) != 1 or runtime_inputs[0].name != "landmarks":
        raise RuntimeError("ONNX model does not expose the required landmarks-only input")
    expected = session.run(["frame_embeddings"], {"landmarks": landmarks})[0]
    return [landmarks], expected


def verify_exported_rknn(output_dir: Path) -> None:
    if RKNN is None:
        raise RuntimeError("rknn-toolkit2 is required for RKNN verification")
    test_inputs, expected = onnx_reference(output_dir / "signlang_det_encoder.onnx")
    name = "signlang_det_encoder.rknn"
    converter = RKNN(verbose=False)
    try:
        _require_success(converter.load_rknn(str(output_dir / name)), "load_rknn")
        _require_success(converter.init_runtime(), "init_runtime")
        outputs = converter.inference(inputs=test_inputs)
        _validate_rknn_outputs(name, outputs, test_inputs, expected)
    finally:
        converter.release()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--pt", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--target-platform", default="rk3588")
    command.add_argument("--validate-only", action="store_true")
    command.add_argument("--verify-rknn-only", action="store_true")
    return command


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parser().parse_args(argv)
    if args.validate_only and args.verify_rknn_only:
        raise ValueError("Only one validation mode can be selected")
    if args.validate_only:
        validate_inputs(args.pt)
    elif args.verify_rknn_only:
        verify_exported_rknn(args.output_dir)
    else:
        convert(args.pt, args.output_dir, args.target_platform)


if __name__ == "__main__":
    main()
