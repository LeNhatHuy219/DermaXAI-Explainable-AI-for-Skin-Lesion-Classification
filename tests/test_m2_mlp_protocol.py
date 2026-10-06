# Kiểm tra protocol của M2-MLP: oracle concepts với mạng MLP.
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Nạp torch qua src.dataset trước sklearn để giữ thứ tự import an toàn trên Windows.
from src import dataset  # noqa: F401
import torch

from src.protocol import load_concept_schema
from experiments import run_m2_mlp
from experiments.run_m2_mlp import evaluate_m2_mlp_test


class M2MLPProtocolTests(unittest.TestCase):
    def setUp(self):
        # Dùng bản sao metadata và checkpoint tạm để kiểm tra pipeline độc lập.
        self.temp = tempfile.TemporaryDirectory(prefix="m2_mlp_protocol_")
        self.path = Path(self.temp.name)
        self.manifest = self.path / "manifest.csv"
        shutil.copyfile(ROOT / "data/manifest.csv", self.manifest)
        shutil.copyfile(ROOT / "data/label_mapping.json", self.path / "label_mapping.json")
        self.schema = load_concept_schema(self.manifest)
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.old_threads)
        self.temp.cleanup()

    def test_oracle_mlp_train_test_round_trip_and_threshold(self):
        # Chạy nhanh hai epochs trên CPU để kiểm tra JSON train/test và ngưỡng đóng băng.
        checkpoint = str(self.path / "mlp.pth")
        with contextlib.redirect_stdout(io.StringIO()):
            run_m2_mlp.main(["--manifest_path", str(self.manifest), "--epochs", "2",
                             "--batch_size", "128", "--device", "cpu", "--checkpoint_path", checkpoint,
                             "--results_path", str(self.path / "mlp_results.json")])
        validation = test = json.loads((self.path / "mlp_results.json").read_text())
        self.assertEqual(test["mode"], "train_validation_test")
        self.assertEqual(test["decision_threshold"], validation["decision_threshold"])
        self.assertEqual(test["concept_schema"], self.schema)
        self.assertEqual(len(test["test_predictions"]), 395)
        for row in test["test_predictions"]:
            self.assertEqual(row["y_pred"], int(row["y_prob"] >= test["decision_threshold"]))

        # Đảo mapping state phải khiến evaluator từ chối checkpoint đã đóng băng schema.
        mapping = copy.deepcopy(self.schema["label_mapping"])
        mapping["pigment_network"]["typical"], mapping["pigment_network"]["atypical"] = (
            mapping["pigment_network"]["atypical"], mapping["pigment_network"]["typical"])
        (self.path / "label_mapping.json").write_text(json.dumps(mapping))
        with self.assertRaises(ValueError):
            evaluate_m2_mlp_test(checkpoint, str(self.manifest), "cpu", save_results=False)


if __name__ == "__main__":
    unittest.main()
