import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace


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

    def test_int8_build_enables_accuracy_guarded_auto_hybrid(self):
        calls = {}
        expected = object()

        class FakeRKNN:
            def __init__(self, verbose=False):
                calls["verbose"] = verbose

            def config(self, **kwargs):
                calls["config"] = kwargs
                return 0

            def load_onnx(self, **kwargs):
                calls["load_onnx"] = kwargs
                return 0

            def build(self, **kwargs):
                calls["build"] = kwargs
                return 0

            def export_rknn(self, path):
                calls["export_rknn"] = path
                return 0

            def init_runtime(self):
                return 0

            def inference(self, inputs):
                return [object()]

            def release(self):
                calls["released"] = True

        original = self.module.RKNN
        original_validator = self.module._validate_rknn_outputs
        self.module.RKNN = FakeRKNN
        self.module._validate_rknn_outputs = lambda *args: calls.update(validated=args)
        try:
            self.module._build_one_rknn(
                Path("encoder.onnx"),
                Path("encoder.int8.rknn"),
                "rk3588",
                Path("dataset.txt"),
                [object(), object()],
                expected,
            )
        finally:
            self.module.RKNN = original
            self.module._validate_rknn_outputs = original_validator

        self.assertEqual(
            calls["config"],
            {"target_platform": "rk3588", "auto_hybrid_cos_thresh": 0.99},
        )
        self.assertEqual(
            calls["build"],
            {
                "do_quantization": True,
                "dataset": "dataset.txt",
                "auto_hybrid": True,
            },
        )
        self.assertTrue(calls["released"])
        self.assertTrue(calls["validated"][-1])

    def test_fp_build_does_not_enable_hybrid_quantization(self):
        calls = {}

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
                return [object()]

            def release(self):
                calls["released"] = True

        original = self.module.RKNN
        original_validator = self.module._validate_rknn_outputs
        self.module.RKNN = FakeRKNN
        self.module._validate_rknn_outputs = lambda *args: None
        try:
            self.module._build_one_rknn(
                Path("encoder.onnx"),
                Path("encoder.rknn"),
                "rk3588",
                None,
                [object(), object()],
                object(),
            )
        finally:
            self.module.RKNN = original
            self.module._validate_rknn_outputs = original_validator

        self.assertEqual(calls["config"], {"target_platform": "rk3588"})
        self.assertEqual(calls["build"], {"do_quantization": False})
        self.assertTrue(calls["released"])


class ModelManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        script = Path(__file__).parents[1] / ".github" / "scripts" / "kaggle_cd.py"
        spec = importlib.util.spec_from_file_location("kaggle_cd_under_test", script)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_hybrid_int8_manifest_declares_float16_output(self):
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

        contract = manifest["models"]["signlang_det_encoder.int8.rknn"]["io_contract"]
        features = contract["inputs"]["features"]
        lengths = contract["inputs"]["lengths"]
        output = contract["outputs"]["frame_embeddings"]
        self.assertEqual(features["dtype"], "int8")
        self.assertIn("quantization_parameters", features)
        self.assertEqual(lengths["dtype"], "int32")
        self.assertEqual(lengths["shape"], [1])
        self.assertEqual(output["dtype"], "float16")
        self.assertNotIn("quantization_parameters", output)


if __name__ == "__main__":
    unittest.main()
