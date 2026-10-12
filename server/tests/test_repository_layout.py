"""Standard-library checks for navigability and committed experiment schemas."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TestShowcaseLayout(unittest.TestCase):
    def test_source_entrypoints(self):
        paths = [
            "distillation/train_early9_lora.py",
            "distillation/train_early9_p0_matched_pair.py",
            "distillation/early9_p0_matched_pair_core.py",
            "evaluation/eval_early9_matched_pair_genimage.py",
            "export/early9_pair_deployment_v1/export_pair.py",
            "export/early9_pair_deployment_v1/convert_pair_fp16.py",
            "docs/RESULTS.md",
            "docs/SOURCE_FILE_MANIFEST.csv",
        ]
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / path).is_file())

    def test_original_evidence_algorithm_symbols(self):
        path = ROOT / "distillation" / "early9_p0_matched_pair_core.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        symbols = {
            n.name for n in module.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for symbol in ("masks", "masked_scores", "edge_loss", "compute_loss", "fake_mass_stats"):
            self.assertIn(symbol, symbols)

    def test_experiment_reports_exist(self):
        base = ROOT / "distillation" / "p0_matched_pair_20k_v1"
        a = json.loads((base / "R_BCE" / "summary.json").read_text())
        b = json.loads((base / "R_Debias" / "summary.json").read_text())
        self.assertEqual(a["final_step"], b["final_step"])
        self.assertEqual(a["final_step"], 2625)
        self.assertEqual(a["shared_init_sha256"], b["shared_init_sha256"])
        self.assertEqual(a["signature"]["train_count"], 20000)
        self.assertEqual(b["signature"]["val_count"], 600)

    def test_readme_highlights_core_source(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("distillation/early9_p0_matched_pair_core.py", text)
        self.assertIn("export/early9_pair_deployment_v1/export_pair.py", text)


if __name__ == "__main__":
    unittest.main()
