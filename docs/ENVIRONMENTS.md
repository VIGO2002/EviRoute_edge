# Server and RK3588 environment separation

The repository has two source partitions, each meant to be copied into a **different original working directory**. The internal folders remain unchanged.

## AutoDL server

Historical root: `/root/autodl-tmp/EviRoute_edge/`. Copy the **contents of `server/`**, not the `server/` directory itself, into that root. Training scripts intentionally reference their original checkpoints, dataset manifests and hash guards. Do not change these SHA checks or overwrite old experiments.

From the original server root:

```bash
cd /root/autodl-tmp/EviRoute_edge
python tools/verify_source_manifest.py
python -m unittest discover -s tests -p 'test_early9_p0_matched_pair.py' -v
python distillation/train_early9_p0_matched_pair.py --help
```

Model conversion uses the historically compatible RKNN-Toolkit2 environment. Resulting RKNN models must be transferred separately to the board and verified by SHA256.

## RK3588 device

Historical root: `/root/EviRoute_RK3588/`. Copy the **contents of `rk3588/`**, not the wrapper, to that root. Two original `.rknn` binaries are required in `models/early9_rbce_fp16/` and `models/early9_rdebias_fp16/`. JSON conversion metadata alone is insufficient.

From the original device root (Ubuntu 22.04/aarch64, matching RKNNLite2 runtime):

```bash
cd /root/EviRoute_RK3588
python3 -m unittest discover -s scripts/early9_full_pipeline_v1 -p 'test_*.py' -v
python3 -m unittest discover -s scripts/early9_async_pipeline_v1 -p 'test_*.py' -v
python3 -m unittest discover -s scripts/eviroute_edge_ui_v1 -p 'test_*.py' -v
bash scripts/eviroute_edge_ui_v1/start_server.sh
# With the graphical session ready, in a second terminal:
bash scripts/eviroute_edge_ui_v1/launch_kiosk.sh topeet
```

**Dependency:** `scripts/early9_async_pipeline_v1/run_async.py` imports `run_images.py` from sibling `scripts/early9_full_pipeline_v1/`. Moving either separately breaks the expected dependency.

The offline HTTP service defaults to 127.0.0.1:8765. User-supplied images and job records belong in the board's private `eviroute_edge_data/`, not in public Git.

## GitHub checkout integrity checks

```bash
python server/tools/verify_source_manifest.py
(cd rk3588 && sha256sum -c docs/SOURCE_SHA256.txt)
python -m unittest discover -s tests -p 'test_layout.py' -v
```

These checks are static or source identity checks, not tests of actual RK3588 inference.
