#!/usr/bin/env python3
"""On-board RK3588 numerical alignment and steady-state NPU inference benchmark.

Uses original Early-9 RGB tensors and original P0 patch-score references.
Requires numpy and rknn-toolkit-lite2 (rknnlite.api) on aarch64 RK3588.
No GPU/PC simulator; preloaded input benchmark excludes decode/resize/disk I/O.
"""
import argparse
import hashlib
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pack_rgb_to_nchw(x):
    """Matches the exact AutoDL exporter mapping (includes memory copy)."""
    if x.shape != (1, 3, 224, 224) or x.dtype != np.float32:
        raise ValueError(f'Expected RGB (1,3,224,224) float32, got {x.shape} {x.dtype}')
    packed = np.ascontiguousarray(
        x.reshape(1, 3, 14, 16, 14, 16)
         .transpose(0, 1, 3, 5, 2, 4)
         .reshape(1, 768, 1, 196)
    )
    if packed.shape != (1, 768, 1, 196) or not packed.flags.c_contiguous:
        raise RuntimeError('Bad packed layout')
    return packed


def top10_info(a):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if a.size != 196 or not np.isfinite(a).all():
        raise RuntimeError('Output must contain 196 finite patch scores')
    ids = np.argsort(-a, kind='stable')[:10]
    return float(a[ids].mean()), set(map(int, ids))


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 0 else float('nan')


def read_temp_c():
    result = []
    for p in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp')):
        try:
            value = float(p.read_text().strip())
            temp = value / 1000.0 if value > 1000.0 else value
            label_path = p.parent / 'type'
            label = label_path.read_text().strip() if label_path.exists() else p.parent.name
            if -40 < temp < 150:
                result.append({'zone': label, 'temp_c': round(temp, 2)})
        except (OSError, ValueError):
            continue
    return result


def read_vmrss_kib():
    try:
        content = Path('/proc/self/status').read_text()
        match = re.search(r'^VmRSS:\s+(\d+)\s+kB', content, flags=re.MULTILINE)
        return int(match.group(1)) if match else None
    except OSError:
        return None


def load_cases(folder):
    folder = Path(folder)
    files = sorted(folder.glob('case_*_input.npy'))
    if len(files) < 4:
        raise RuntimeError(f'Need >=4 case_*_input.npy files; got {len(files)} in {folder}')
    cases = []
    for inp_path in files:
        ref_path = inp_path.with_name(inp_path.name.replace('_input.npy', '_pytorch_scores.npy'))
        if not ref_path.exists():
            raise FileNotFoundError(f'Missing matching reference: {ref_path}')
        inp = np.load(inp_path, allow_pickle=False)
        ref = np.load(ref_path, allow_pickle=False)
        if inp.shape != (1, 3, 224, 224) or inp.dtype != np.float32:
            raise ValueError(f'Unexpected input shape/dtype: {inp_path} {inp.shape} {inp.dtype}')
        if ref.size != 196 or not np.isfinite(ref).all() or not np.isfinite(inp).all():
            raise ValueError(f'Invalid case/reference: {inp_path}')
        cases.append((inp_path.stem.replace('_input', ''), np.ascontiguousarray(inp), np.asarray(ref, dtype=np.float32).reshape(-1)))
    return cases


def inference(lite, inp):
    outputs = lite.inference(inputs=[inp], data_format=['nchw'])
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1 or outputs[0] is None:
        raise RuntimeError('RKNNLite inference returned no valid output')
    a = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if a.size != 196 or not np.isfinite(a).all():
        raise RuntimeError(f'Bad NPU output: shape={np.shape(outputs[0])}')
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--cases-dir', type=Path, required=True)
    p.add_argument('--core', default='0', choices=['auto', '0', '1', '2', '0_1_2'])
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--runs', type=int, default=200)
    p.add_argument('--out', type=Path, default=Path('early9_external_patchify_fast_board_report.json'))
    p.add_argument('--expect-sha256', default=None, help='Optional SHA256 from conversion host')
    args = p.parse_args()
    if args.warmup < 0 or args.runs < 30:
        p.error('Require --warmup >=0 and --runs >=30')
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError(f'Board only: machine={platform.machine()} (no PC simulator)')
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model_hash = sha256_file(args.model)
    print('BOARD / RKNN MODEL')
    print('Architecture:', platform.machine())
    print('Model:', args.model, f'({args.model.stat().st_size / 1024**2:.2f} MiB)')
    print('SHA256:', model_hash)
    if args.expect_sha256 and model_hash.lower() != args.expect_sha256.lower():
        raise RuntimeError('Model file SHA256 mismatch; abort')
    cases = load_cases(args.cases_dir)
    print('Validation cases:', len(cases))
    from rknnlite.api import RKNNLite
    mask_map = {
        'auto': 'NPU_CORE_AUTO', '0': 'NPU_CORE_0',
        '1': 'NPU_CORE_1', '2': 'NPU_CORE_2',
        '0_1_2': 'NPU_CORE_0_1_2',
    }
    const_name = mask_map[args.core]
    if not hasattr(RKNNLite, const_name):
        raise RuntimeError(f'RKNNLite missing core mask: {const_name}')
    lite = RKNNLite(verbose=False)
    report = {'model': str(args.model), 'model_sha256': model_hash, 'core': args.core, 'num_reference_cases': len(cases), 'runs': args.runs, 'warmup': args.warmup}
    try:
        tic = time.perf_counter()
        ret = lite.load_rknn(str(args.model))
        if ret != 0:
            raise RuntimeError(f'load_rknn failed, return={ret}')
        report['load_seconds'] = round(time.perf_counter() - tic, 4)
        print('load_rknn: PASS')
        tic = time.perf_counter()
        ret = lite.init_runtime(core_mask=getattr(RKNNLite, const_name))
        if ret != 0:
            raise RuntimeError(f'init_runtime failed, return={ret}')
        report['init_seconds'] = round(time.perf_counter() - tic, 4)
        print('init_runtime: PASS', 'core=', args.core)
        print('\nNUMERICAL AUDIT (real RK3588 NPU):')
        comparisons = []
        for case, inp, ref in cases:
            y = inference(lite, pack_rgb_to_nchw(inp))
            ref_logit, ref_top = top10_info(ref)
            logit, y_top = top10_info(y)
            metrics = {
                'case': case,
                'cosine': cosine(ref, y),
                'mae': float(np.abs(y.astype(np.float64)-ref).mean()),
                'max_abs': float(np.max(np.abs(y.astype(np.float64)-ref))),
                'top10_overlap': len(ref_top & y_top),
                'logit_abs_delta': abs(ref_logit - logit),
                'pred_match_at_zero': bool((ref_logit >= 0) == (logit >= 0)),
            }
            comparisons.append(metrics)
            print(f"{case:10s} cos={metrics['cosine']:.8f} max_abs={metrics['max_abs']:.5f} "
                  f"Top10={metrics['top10_overlap']}/10 logit_err={metrics['logit_abs_delta']:.5f} "
                  f"pred_match={metrics['pred_match_at_zero']}")
        summary = {
            'min_cosine': min(x['cosine'] for x in comparisons),
            'max_abs': max(x['max_abs'] for x in comparisons),
            'mean_mae': float(np.mean([x['mae'] for x in comparisons])),
            'max_logit_error': max(x['logit_abs_delta'] for x in comparisons),
            'mean_top10_overlap': float(np.mean([x['top10_overlap'] for x in comparisons])),
            'prediction_matches': sum(x['pred_match_at_zero'] for x in comparisons),
            'num_samples': len(comparisons),
        }
        report['numerical'] = summary
        report['individual'] = comparisons
        print('Numerical summary:', json.dumps(summary, ensure_ascii=False, indent=2))
        if not np.isfinite(summary['min_cosine']):
            raise RuntimeError('Invalid numerical comparisons')

        # Warmup excludes model loading and compile/initialization effects.
        inputs = [x[1] for x in cases]
        for k in range(args.warmup):
            inference(lite, pack_rgb_to_nchw(inputs[k % len(inputs)]))
        report['temps_before'] = read_temp_c()
        print(f'\nBENCHMARK: {args.runs} RGB->packed + RKNN + Top10 calls; RGB tensors in RAM')
        durations_ms = []
        pack_ms, infer_ms, post_ms = [], [], []
        tic = time.perf_counter()
        for k in range(args.runs):
            start = time.perf_counter()
            packed = pack_rgb_to_nchw(inputs[k % len(inputs)])
            end_packing = time.perf_counter()
            out = inference(lite, packed)
            end_inference = time.perf_counter()
            _ = top10_info(out)
            end_post = time.perf_counter()
            pack_ms.append((end_packing-start)*1000.0)
            infer_ms.append((end_inference-end_packing)*1000.0)
            post_ms.append((end_post-end_inference)*1000.0)
            durations_ms.append((end_post-start)*1000.0)
        wall = time.perf_counter() - tic
        v = np.asarray(durations_ms, dtype=np.float64)
        bench = {
            'measurement': 'sequential RGB->external CPU patchify + NPU inference API + CPU top10; 12 RGB tensors preloaded in RAM; excludes image decode/resize, disk/network I/O and UI',
            'total_s': float(wall),
            'fps': float(args.runs / wall),
            'mean_ms': float(v.mean()),
            'median_ms': float(np.median(v)),
            'p90_ms': float(np.percentile(v, 90)),
            'p95_ms': float(np.percentile(v, 95)),
            'min_ms': float(v.min()),
            'max_ms': float(v.max()),
        }
        bench['phase_mean_ms'] = {
            'external_cpu_patchify': float(np.mean(pack_ms)),
            'rknn_inference_api': float(np.mean(infer_ms)),
            'cpu_top10': float(np.mean(post_ms)),
        }
        report['benchmark'] = bench
        report['temps_after'] = read_temp_c()
        report['process_vmrss_kib'] = read_vmrss_kib()
        print('Benchmark result:', json.dumps(bench, ensure_ascii=False, indent=2))
        print('Process VmRSS (KiB):', report['process_vmrss_kib'])
        print('Temperature before:', report['temps_before'])
        print('Temperature after: ', report['temps_after'])
        print('BOARD INFERENCE TEST: COMPLETE (numeric metrics require review)')
    finally:
        try:
            lite.release()
        except Exception as e:
            print('WARNING: RKNNLite release failed:', e, file=sys.stderr)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print('Saved report:', args.out)


if __name__ == '__main__':
    main()
