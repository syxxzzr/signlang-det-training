import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest import mock


class RknnBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).parents[1] / ".github" / "scripts" / "convert_models.py"
        spec = importlib.util.spec_from_file_location("convert_models_under_test", script)
        cls.module = importlib.util.module_from_spec(spec)
        module_names = (
            "numpy",
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "rknn",
            "rknn.api",
        )
        originals = {name: sys.modules.get(name) for name in module_names}
        numpy_stub = ModuleType("numpy")
        torch_stub = ModuleType("torch")
        nn_stub = ModuleType("torch.nn")
        functional_stub = ModuleType("torch.nn.functional")
        rknn_stub = ModuleType("rknn")
        rknn_api_stub = ModuleType("rknn.api")

        class Module:
            pass

        nn_stub.Module = Module
        nn_stub.functional = functional_stub
        torch_stub.nn = nn_stub
        rknn_api_stub.RKNN = None
        rknn_stub.api = rknn_api_stub
        sys.modules.update(
            {
                "numpy": numpy_stub,
                "torch": torch_stub,
                "torch.nn": nn_stub,
                "torch.nn.functional": functional_stub,
                "rknn": rknn_stub,
                "rknn.api": rknn_api_stub,
            }
        )
        try:
            spec.loader.exec_module(cls.module)
        finally:
            for name, original in originals.items():
                if original is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original

    def test_build_uses_the_standard_rknn_configuration(self):
        calls = {}

        self.assertTrue(hasattr(self.module, "_build_rknn"))

        class FakeRKNN:
            def __init__(self, verbose=False):
                pass

            def config(self, **kwargs):
                calls["config"] = kwargs
                return 0

            def load_onnx(self, **kwargs):
                return 0

            def build(self, **kwargs):
                calls["build"] = kwargs
                return 0

            def export_rknn(self, path):
                return 0

            def init_runtime(self):
                return 0

            def inference(self, inputs):
                calls["inference_inputs"] = inputs
                return [object()]

            def release(self):
                calls["released"] = True

        original = self.module.RKNN
        original_validator = self.module._validate_rknn_outputs
        self.module.RKNN = FakeRKNN
        self.module._validate_rknn_outputs = lambda *args: None
        try:
            self.module._build_rknn(
                Path("encoder.onnx"),
                Path("encoder.rknn"),
                "rk3588",
                [object()],
                object(),
            )
        finally:
            self.module.RKNN = original
            self.module._validate_rknn_outputs = original_validator

        self.assertEqual(calls["config"], {"target_platform": "rk3588"})
        self.assertEqual(calls["build"], {"do_quantization": False})
        self.assertEqual(len(calls["inference_inputs"]), 1)
        self.assertTrue(calls["released"])

    def test_convert_creates_one_rknn_model(self):
        calls = {}
        self.assertEqual(
            list(inspect.signature(self.module.convert).parameters),
            ["pt_path", "output_dir", "target_platform"],
        )
        original_export = self.module.export_onnx
        original_build = getattr(self.module, "build_rknn_model", None)
        self.module.export_onnx = lambda pt, onnx: (
            calls.update(export=(pt, onnx)) or ("landmarks", "expected")
        )
        self.module.build_rknn_model = lambda *args: calls.update(build=args)
        try:
            outputs = self.module.convert(
                Path("encoder.pt"), Path("outputs"), "rk3588"
            )
        finally:
            self.module.export_onnx = original_export
            if original_build is None:
                delattr(self.module, "build_rknn_model")
            else:
                self.module.build_rknn_model = original_build

        self.assertEqual(
            calls["export"],
            (Path("encoder.pt"), Path("outputs/signlang_det_encoder.onnx")),
        )
        self.assertEqual(
            calls["build"],
            (
                Path("outputs/signlang_det_encoder.onnx"),
                Path("outputs/signlang_det_encoder.rknn"),
                "rk3588",
                ["landmarks"],
                "expected",
            ),
        )
        self.assertEqual(
            outputs,
            [
                Path("outputs/signlang_det_encoder.onnx"),
                Path("outputs/signlang_det_encoder.rknn"),
            ],
        )

    def test_cli_requires_only_model_and_output_paths(self):
        options = {
            option
            for action in self.module.parser()._actions
            for option in action.option_strings
        }
        self.assertEqual(
            options,
            {
                "-h",
                "--help",
                "--pt",
                "--output-dir",
                "--target-platform",
                "--validate-only",
                "--verify-rknn-only",
            },
        )

    def test_deployment_wrapper_derives_padding_mask_from_landmarks(self):
        calls = {}

        class FakeValidMask:
            def to(self, dtype):
                calls["length_dtype"] = dtype
                return "lengths"

        class FakePaddingMask:
            def __invert__(self):
                calls["inverted"] = True
                return "valid-mask"

        class FakeLandmarks:
            def __lt__(self, value):
                calls["padding_threshold"] = value
                return "padding-comparison"

        class FakeEncoder:
            def __call__(self, landmarks, lengths):
                calls["encoder"] = (landmarks, lengths)
                return "frame-embeddings"

        original_all = getattr(self.module.torch, "all", None)
        original_sum = getattr(self.module.torch, "sum", None)
        original_int32 = getattr(self.module.torch, "int32", None)
        self.module.torch.int32 = "int32"
        self.module.torch.all = lambda value, dim: calls.update(
            reduction=(value, dim)
        ) or FakePaddingMask()
        self.module.torch.sum = lambda value, dim: calls.update(
            summation=(value, dim)
        ) or FakeValidMask()
        try:
            landmarks = FakeLandmarks()
            wrapper = self.module.DeploymentEncoder(FakeEncoder())
            output = wrapper.forward(landmarks)
        finally:
            if original_all is None:
                delattr(self.module.torch, "all")
            else:
                self.module.torch.all = original_all
            if original_sum is None:
                delattr(self.module.torch, "sum")
            else:
                self.module.torch.sum = original_sum
            if original_int32 is None:
                delattr(self.module.torch, "int32")
            else:
                self.module.torch.int32 = original_int32

        self.assertEqual(calls["padding_threshold"], -50.0)
        self.assertEqual(calls["reduction"], ("padding-comparison", -1))
        self.assertTrue(calls["inverted"])
        self.assertEqual(calls["summation"], ("valid-mask", 1))
        self.assertEqual(calls["length_dtype"], "int32")
        self.assertEqual(calls["encoder"], (landmarks, "lengths"))
        self.assertEqual(output, "frame-embeddings")


class ModelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).parents[1] / ".github" / "scripts" / "kaggle_cd.py"
        spec = importlib.util.spec_from_file_location("kaggle_cd_under_test", script)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_manifest_describes_the_three_published_models(self):
        payload = {
            "preprocessing": "hand168-temporal",
            "encoder_fingerprint": "fingerprint",
            "model_config": {"embedding_dim": 128},
            "input_contract": {"dtype": "float32", "shape": ["B", 64, 168]},
            "output_contract": {"dtype": "float32", "shape": ["B", 64, 128]},
        }
        state = {
            "tag": "v-test",
            "git_sha": "deadbeef",
            "kaggle_kernel": "owner/kernel",
            "kaggle_version": 1,
            "kaggle_url": "https://example.invalid/kernel",
            "rknn_target_platform": "rk3588",
        }
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory)
            pt_path = assets / "signlang_det_encoder.pt"
            pt_path.write_bytes(b"test checkpoint")
            for name in self.module.RELEASE_ASSET_FILES:
                path = assets / name
                if path != pt_path:
                    path.write_bytes(name.encode())
            torch_stub = ModuleType("torch")
            torch_stub.load = lambda *args, **kwargs: payload
            original_torch = sys.modules.get("torch")
            sys.modules["torch"] = torch_stub
            try:
                manifest_path = self.module.create_model_manifest(
                    pt_path,
                    assets,
                    state,
                    SimpleNamespace(rknn_target_platform="rk3588"),
                    "converter-sha256",
                )
            finally:
                if original_torch is None:
                    sys.modules.pop("torch", None)
                else:
                    sys.modules["torch"] = original_torch
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        pt_contract = manifest["models"]["signlang_det_encoder.pt"]["io_contract"]
        onnx_contract = manifest["models"]["signlang_det_encoder.onnx"]["io_contract"]
        rknn_contract = manifest["models"]["signlang_det_encoder.rknn"]["io_contract"]
        self.assertEqual(set(pt_contract["inputs"]), {"features", "lengths"})
        self.assertEqual(set(onnx_contract["inputs"]), {"landmarks"})
        self.assertEqual(onnx_contract["inputs"]["landmarks"]["dtype"], "float32")
        self.assertEqual(rknn_contract, onnx_contract)
        self.assertEqual(
            set(manifest["models"]),
            {
                "signlang_det_encoder.pt",
                "signlang_det_encoder.onnx",
                "signlang_det_encoder.rknn",
            },
        )
        self.assertTrue(all(
            set(model) == {"format", "io_contract", "bytes", "sha256"}
            for model in manifest["models"].values()
        ))
        self.assertEqual(
            manifest["artifacts"],
            {
                "pytorch_weights": "signlang_det_encoder.pt",
                "onnx_graph": "signlang_det_encoder.onnx",
                "rknn_graph": "signlang_det_encoder.rknn",
                "tokenizer": None,
            },
        )

    def test_release_conversion_uses_the_model_only_cli(self):
        state = {
            "tag": "v-test",
            "git_sha": "deadbeef",
            "kaggle_kernel": "owner/kernel",
            "kaggle_version": 1,
            "kaggle_url": "https://example.invalid/kernel",
            "rknn_target_platform": "rk3588",
        }
        commands = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "output"
            assets = root / "assets"
            for name in self.module.NOTEBOOK_OUTPUT_FILES:
                path = output_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())

            def fake_check_output(command, text=False):
                return "{}" if text else b"tagged converter"

            def fake_run(command):
                commands.append(command)
                if "--validate-only" not in command and "--verify-rknn-only" not in command:
                    output_dir = Path(command[command.index("--output-dir") + 1])
                    for name in (
                        "signlang_det_encoder.onnx",
                        "signlang_det_encoder.rknn",
                    ):
                        (output_dir / name).write_bytes(name.encode())

            def fake_manifest(pt_path, asset_dir, current_state, config, digest):
                path = asset_dir / "model-manifest.json"
                path.write_text("{}\n", encoding="utf-8")
                return path

            with (
                mock.patch.object(self.module.subprocess, "check_output", fake_check_output),
                mock.patch.object(self.module, "run", fake_run),
                mock.patch.object(self.module, "create_model_manifest", fake_manifest),
            ):
                result = self.module.create_release_assets(
                    output_root,
                    assets,
                    state,
                    SimpleNamespace(rknn_target_platform="rk3588"),
                )

            current_converter = str(Path(self.module.__file__).with_name("convert_models.py"))
            pt_path = str(assets / "signlang_det_encoder.pt")
            output_dir = str(assets)
            common = [
                "--pt",
                pt_path,
                "--output-dir",
                output_dir,
                "--target-platform",
                "rk3588",
            ]
            self.assertEqual(
                commands[0],
                [sys.executable, current_converter, *common, "--validate-only"],
            )
            self.assertEqual(commands[1][0], sys.executable)
            self.assertEqual(commands[1][2:], common)
            self.assertEqual(
                commands[2],
                [sys.executable, current_converter, *common, "--verify-rknn-only"],
            )
            self.assertEqual(len(result), 6)


if __name__ == "__main__":
    unittest.main()
