import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestWorkspaceSplit(unittest.TestCase):
    def test_critical_paths(self):
        for relative in (
            "server/distillation/early9_p0_matched_pair_core.py",
            "server/export/early9_pair_deployment_v1/export_pair.py",
            "server/docs/SOURCE_FILE_MANIFEST.csv",
            "rk3588/scripts/early9_full_pipeline_v1/run_images.py",
            "rk3588/scripts/early9_async_pipeline_v1/run_async.py",
            "rk3588/scripts/eviroute_edge_ui_v1/server.py",
            "rk3588/scripts/eviroute_edge_ui_v1/static/index.html",
            "rk3588/docs/SOURCE_SHA256.txt",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_async_import_retained(self):
        code = (ROOT / "rk3588/scripts/early9_async_pipeline_v1/run_async.py").read_text()
        self.assertIn("import run_images as old", code)
        self.assertIn("early9_full_pipeline_v1", code)

if __name__ == "__main__":
    unittest.main()
