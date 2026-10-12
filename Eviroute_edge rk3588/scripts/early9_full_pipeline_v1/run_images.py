#!/usr/bin/env python3
"""RK3588 real JPEG/PNG inference for matched Early-9 R-BCE and R-Debias.

Full sequential pipeline: file I/O, Pillow decode and bicubic resize, RGB [0,1]
float32, externally packed 16x16 patches, RKNNLite NPU, CPU Top10 and optional
Pillow visualization. Outputs per-image JSONL and a run summary JSON.

No ImageNet normalization; no model calibration or threshold retuning.
"""
from __future__ import annotations

import argparse
import io
import hashlib
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_SHA256 = {
    'R_BCE': '808cecbf53a60d3071d622a32792f7cce782a22115c97b2ead9d7cbda75cf597',
    'R_Debias': '40db8e07ef6fe537c913d19bdbbb0fe01ec0c1f3a1fba14d1e86a6ae6e4a251c',
}
MODEL_DIRECTORY = {'R_BCE': 'early9_rbce_fp16', 'R_Debias': 'early9_rdebias_fp16'}
CORE_MASK = {'auto': 'NPU_CORE_AUTO', '0': 'NPU_CORE_0', '1': 'NPU_CORE_1',
             '2': 'NPU_CORE_2', '0_1_2': 'NPU_CORE_0_1_2'}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, 'Resampling') else Image.BICUBIC


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_images(input_path: Path, recursive=True) -> tuple[list[Path], Path]:
    p = Path(input_path).expanduser().resolve()
    if p.is_file():
        if p.suffix.lower() not in EXTENSIONS:
            raise ValueError(f'Not a supported image: {p}')
        return [p], p.parent
    if not p.is_dir():
        raise FileNotFoundError(f'Image input does not exist: {p}')
    iterator = p.rglob('*') if recursive else p.glob('*')
    images = sorted((x for x in iterator if x.is_file() and x.suffix.lower() in EXTENSIONS),
                    key=lambda x: str(x.relative_to(p)).lower())
    if not images:
        raise FileNotFoundError(f'No JPEG/PNG/etc. images in: {p}')
    return images, p


def read_rgb224(path: Path, respect_exif=False) -> tuple[np.ndarray, tuple[int, int], str]:
    # Read once; checksum of the original encoded file enables exact same-input auditing.
    raw = Path(path).read_bytes()
    input_sha256 = hashlib.sha256(raw).hexdigest()
    with Image.open(io.BytesIO(raw)) as im:
        if respect_exif:
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
        size = im.size
        # Exact old evaluator protocol: RGB, unconditional bicubic 224x224 resize.
        # EXIF handling is OFF by default to match existing PyTorch evaluation.
        rgb = im.convert('RGB').resize((224, 224), BICUBIC)
        array = np.asarray(rgb, dtype=np.uint8).copy()
    if array.shape != (224, 224, 3):
        raise RuntimeError(f'Invalid RGB 224 array for {path}: {array.shape}')
    return array, size, input_sha256


def rgb_uint8_to_packed(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.shape != (224, 224, 3) or rgb.dtype != np.uint8:
        raise ValueError(f'Expected RGB uint8 [224,224,3], got {rgb.shape} / {rgb.dtype}')
    nchw = np.ascontiguousarray((rgb.astype(np.float32) / np.float32(255.0)).transpose(2, 0, 1)[None])
    x = np.ascontiguousarray(
        nchw.reshape(1, 3, 14, 16, 14, 16)
            .transpose(0, 1, 3, 5, 2, 4)
            .reshape(1, 768, 1, 196)
    )
    assert x.shape == (1, 768, 1, 196) and x.dtype == np.float32
    return x


def logit_and_top10(scores: np.ndarray) -> tuple[float, np.ndarray]:
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    if a.size != 196 or not np.isfinite(a).all():
        raise ValueError('Expected 196 finite scores')
    ids = np.argsort(-a, kind='stable')[:10]
    return float(a[ids].mean()), ids


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def evidence_stats(indices: np.ndarray) -> dict:
    ids = np.asarray(indices, dtype=np.int64)
    if len(ids) != 10 or np.unique(ids).size != 10 or np.any((ids < 0) | (ids > 195)):
        raise ValueError('Need 10 distinct patch indices in 0..195')
    r = ids // 14
    c = ids % 14
    b1 = (r == 0) | (r == 13) | (c == 0) | (c == 13)
    b2 = (r < 2) | (r > 11) | (c < 2) | (c > 11)
    return {'b1_count': int(b1.sum()), 'b2_only_count': int((b2 & ~b1).sum()),
            'interior_count': int((~b2).sum())}


def create_visual(rgb: np.ndarray, ids: np.ndarray, mode: str) -> Image.Image:
    """Overlay top ten *fake-support* scores on the actual 224x224 model input."""
    source = Image.fromarray(rgb, mode='RGB')
    overlay = source.convert('RGBA')
    draw = ImageDraw.Draw(overlay, 'RGBA')
    # Transparent fill, visible outline, rank is optional informative text.
    for rank, idx in enumerate(ids):
        rr, cc = divmod(int(idx), 14)
        x0, y0 = cc * 16, rr * 16
        rect = (x0, y0, x0 + 15, y0 + 15)
        color = (240, 60, 44, 240) if rank == 0 else (250, 155, 24, 240)
        draw.rectangle(rect, fill=(250, 90, 40, 68), outline=color, width=2)
    out = overlay.convert('RGB')
    if mode == 'comparison':
        canvas = Image.new('RGB', (448, 224))
        canvas.paste(source, (0, 0))
        canvas.paste(out, (224, 0))
        return canvas
    if mode == 'overlay':
        return out
    raise ValueError(f'Unsupported visualization mode: {mode}')


def infer_scores(lite, packed: np.ndarray) -> np.ndarray:
    outputs = lite.inference(inputs=[packed], data_format=['nchw'])
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 1 or outputs[0] is None:
        raise RuntimeError('Unexpected RKNN output collection')
    scores = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    if scores.size != 196 or not np.isfinite(scores).all():
        raise RuntimeError(f'Expected exactly 196 finite patch scores, got {scores.shape}')
    return scores


def summarize_ms(values):
    arr = np.asarray(values, dtype=np.float64)
    return {'mean': float(arr.mean()), 'p50': float(np.median(arr)),
            'p90': float(np.percentile(arr, 90)), 'p95': float(np.percentile(arr, 95)),
            'min': float(arr.min()), 'max': float(arr.max())}


def get_thermal():
    temps = {}
    for p in sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp')):
        try:
            value = float(p.read_text().strip())
            c = value / 1000 if value > 1000 else value
            if not -40 < c < 150:
                continue
            label = (p.parent / 'type').read_text().strip()
            temps[label] = round(c, 2)
        except (OSError, ValueError):
            pass
    return temps


def vmrss_kib():
    try:
        m = re.search(r'^VmRSS:\s+(\d+)\s+kB', Path('/proc/self/status').read_text(), flags=re.M)
        return int(m.group(1)) if m else None
    except OSError:
        return None


def model_path_for_arm(project_root, arm):
    model_dir = Path(project_root) / 'models' / MODEL_DIRECTORY[arm]
    matches = sorted(model_dir.glob('*.rknn'))
    if len(matches) != 1:
        raise RuntimeError(f'{arm}: expected exactly one .rknn under {model_dir}, found {len(matches)}')
    return matches[0]


def execute(args, lite_class=None):
    if not args.allow_other_arch and platform.machine().lower() not in ('aarch64', 'arm64'):
        raise RuntimeError('Board application requires real aarch64 RK3588 (no simulator)')
    model = Path(args.model) if args.model else model_path_for_arm(args.project_root, args.arm)
    model = model.expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    digest = sha256_file(model)
    if digest.lower() != MODEL_SHA256[args.arm]:
        raise RuntimeError(f'{args.arm} model SHA256 mismatch: {digest} != {MODEL_SHA256[args.arm]}')
    files, image_root = collect_images(args.input, recursive=not args.non_recursive)
    if args.max_images > 0:
        files = files[:args.max_images]
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing output directory: {out_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_visual != 'none':
        (out_dir / 'visuals').mkdir(exist_ok=True)
    if lite_class is None:
        from rknnlite.api import RKNNLite
        lite_class = RKNNLite
    mask = CORE_MASK[args.core]
    if not hasattr(lite_class, mask):
        raise RuntimeError(f'RKNNLite version does not support {mask}')
    lite = lite_class(verbose=False)
    records = []
    time_keys = ('decode_resize_ms', 'cpu_patchify_ms', 'rknn_inference_ms',
                 'cpu_top10_ms', 'visual_write_ms', 'full_image_ms')
    readings = {k: [] for k in time_keys}
    try:
        code = lite.load_rknn(str(model))
        if code != 0:
            raise RuntimeError(f'load_rknn failed: {code}')
        code = lite.init_runtime(core_mask=getattr(lite_class, mask))
        if code != 0:
            raise RuntimeError(f'init_runtime failed: {code}')
        if args.warmup:
            rgb_first, _, _ = read_rgb224(files[0], respect_exif=args.respect_exif)
            packed_first = rgb_uint8_to_packed(rgb_first)
            for _ in range(args.warmup):
                infer_scores(lite, packed_first)
        start_of_run = time.perf_counter()
        before_temp = get_thermal()
        print(f'RK3588 READY: {args.arm}, model SHA256={digest}, images={len(files)}', flush=True)
        with (out_dir / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
            for i, img_path in enumerate(files):
                t0 = time.perf_counter()
                rgb, original_size, input_sha256 = read_rgb224(img_path, respect_exif=args.respect_exif)
                t1 = time.perf_counter()
                packed = rgb_uint8_to_packed(rgb)
                t2 = time.perf_counter()
                scores = infer_scores(lite, packed)
                t3 = time.perf_counter()
                logit, ids = logit_and_top10(scores)
                regions = evidence_stats(ids)
                probability = sigmoid(logit)
                t4 = time.perf_counter()
                visual_file = None
                if args.save_visual != 'none':
                    safe_stem = re.sub(r'[^a-zA-Z0-9_-]+', '_', img_path.stem)[:42] or 'image'
                    image_id = hashlib.sha1(str(img_path.relative_to(image_root)).encode()).hexdigest()[:10]
                    visual_file = f'visuals/{i:04d}_{safe_stem}_{image_id}.png'
                    vis = create_visual(rgb, ids, args.save_visual)
                    vis.save(out_dir / visual_file, format='PNG')
                t5 = time.perf_counter()
                relative_path = str(img_path.relative_to(image_root))
                row = {
                    'index': i, 'relative_path': relative_path, 'source_path': str(img_path),
                    'input_sha256': input_sha256,
                    'original_size_wh': list(original_size),
                    'prediction': 'FAKE' if logit >= 0 else 'REAL',
                    'fake_label': int(logit >= 0), 'logit': logit, 'probability_fake': probability,
                    'top10_patch_indices': [int(x) for x in ids],
                    'top10_patch_rc': [[int(x // 14), int(x % 14)] for x in ids],
                    'top10_patch_scores': [float(scores[x]) for x in ids],
                    'top10_region_counts': regions,
                    'visual_file': visual_file,
                    'timings_ms': {
                        'decode_resize_ms': (t1 - t0) * 1000,
                        'cpu_patchify_ms': (t2 - t1) * 1000,
                        'rknn_inference_ms': (t3 - t2) * 1000,
                        'cpu_top10_ms': (t4 - t3) * 1000,
                        'visual_write_ms': (t5 - t4) * 1000,
                        'full_image_ms': (t5 - t0) * 1000,
                    },
                }
                if args.store_all_scores:
                    row['all_196_patch_scores'] = scores.tolist()
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                records.append(row)
                for k in time_keys:
                    readings[k].append(row['timings_ms'][k])
                if (i + 1) % args.progress_every == 0 or i + 1 == len(files):
                    print(f'{args.arm} {i + 1}/{len(files)} | latest={row["timings_ms"]["full_image_ms"]:.2f}ms '
                          f'| logit={logit:+.4f} {row["prediction"]}', flush=True)
        wall_s = time.perf_counter() - start_of_run
        summary = {
            'status': 'complete', 'arm': args.arm,
            'model': str(model), 'model_sha256': digest,
            'core': args.core, 'n_images': len(files),
            'input': str(Path(args.input).resolve()), 'image_sort': 'sorted by relative path',
            'warmup': args.warmup,
            'protocol': {
                'decode': 'Pillow RGB conversion, BICUBIC resize (224,224), EXIF transpose '+('ON' if args.respect_exif else 'OFF'),
                'normalization': 'float32 0..1 (divide by float32(255.0)); NO ImageNet norm',
                'packed_input': '[1,768,1,196] float32, 16x16 external patchify',
                'output': '196 NPU fake-support patch scores; top10 mean; logit>=0 -> FAKE',
                'top10_visual_meaning': 'Highest fake-support patches for BOTH predictions; NOT a verified artifact ground-truth mask',
                'probability_fake_meaning': 'sigmoid(logit) display score, NOT externally calibrated probability',
                'save_visual': args.save_visual,
                'time_includes': 'per-image encoded file read+SHA256+decode+224 resize, CPU patchify, RKNN call, Top10, optional PNG generation/write',
                'time_excludes': 'model load/init, warmup, final summary JSON writing, user-interface/network',
                'not_equivalent_to': 'previous preloaded tensor NPU FPS benchmark',
            },
            'wall_seconds': float(wall_s),
            'full_pipeline_images_per_second': float(len(files) / wall_s),
            'phase_ms': {k: summarize_ms(v) for k, v in readings.items()},
            'temperature_c_before': before_temp, 'temperature_c_after': get_thermal(),
            'process_vmrss_kib_after': vmrss_kib(),
            'prediction_counts': {name: sum(r['prediction'] == name for r in records) for name in ('REAL','FAKE')},
            'output': str(out_dir),
        }
        (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print('FULL IMAGE PIPELINE: COMPLETE', args.arm)
        print(json.dumps({k: summary[k] for k in ('n_images', 'full_pipeline_images_per_second', 'phase_ms', 'prediction_counts')}, ensure_ascii=False, indent=2))
        print('Saved:', out_dir)
        return summary
    finally:
        try:
            lite.release()
        except Exception as e:
            print(f'WARNING: RKNN release failed: {e}', file=sys.stderr)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=sorted(MODEL_SHA256), required=True)
    p.add_argument('--input', type=Path, required=True, help='JPEG/PNG file or directory')
    p.add_argument('--project-root', type=Path, default=Path('/root/EviRoute_RK3588'))
    p.add_argument('--model', type=Path, help='Explicit RKNN model override, still SHA256 checked')
    p.add_argument('--out-dir', type=Path, required=True, help='Must be new or empty; never overwrite old results')
    p.add_argument('--core', choices=list(CORE_MASK), default='0_1_2')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--max-images', type=int, default=0, help='0 means all images')
    p.add_argument('--non-recursive', action='store_true')
    p.add_argument('--save-visual', choices=['none','overlay','comparison'], default='comparison')
    p.add_argument('--store-all-scores', action='store_true', help='Store all 196 NPU scores for each image')
    p.add_argument('--respect-exif', action='store_true', help='Optional EXIF transpose; changes original evaluation protocol')
    p.add_argument('--progress-every', type=int, default=10)
    p.add_argument('--allow-other-arch', action='store_true', help=argparse.SUPPRESS)  # CPU unit-test only, still SHA pinned.
    args = p.parse_args(argv)
    if args.warmup < 0 or args.max_images < 0 or args.progress_every < 1:
        p.error('warmup >=0; max-images >=0; progress-every >=1')
    return args


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
