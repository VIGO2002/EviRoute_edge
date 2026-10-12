#!/usr/bin/env python3
"""EviRoute Edge offline-first embedded browser UI.

Local HTTP / file upload / one-at-a-time job queue / proven run_async.py subprocess.
No external API, CDN, Flask, or cloud communication.

IMPORTANT: default bind 127.0.0.1. Do not expose without authentication.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
import zipfile
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, UnidentifiedImageError

VERSION = '1.0.0'
VALID_ARM = ('R_BCE', 'R_Debias')
VALID_SUFFIX = {'.png', '.jpeg', '.jpg', '.webp'}
MIME = {'.png': 'image/png', '.jpeg': 'image/jpeg', '.jpg': 'image/jpeg', '.webp': 'image/webp'}
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_JOB_IMAGES = 200
FILE_ID = re.compile(r'^[0-9a-f]{32}$')
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
STATIC = Path(__file__).resolve().parent / 'static'
EXPECTED_MODELS = {
    'R_BCE': 'early9_rbce_fp16',
    'R_Debias': 'early9_rdebias_fp16',
}

def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def clean_name(name: str) -> str:
    # A client filename is display metadata only: never use it to choose a directory.
    raw = unicodedata.normalize('NFKC', Path(str(name).replace('\\', '/')).name)
    stem = re.sub(r'[^\w. -]+', '_', Path(raw).stem, flags=re.UNICODE).strip(' ._')[:55] or 'image'
    suffix = Path(raw).suffix.lower()
    if suffix not in VALID_SUFFIX:
        raise ValueError('仅接受 JPEG / PNG / WebP 图片')
    return stem + suffix


def store_json(path: Path, value: dict):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


class UploadStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.lock = threading.Lock()
        for p in self.root.glob('*.json'):
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                image_id = d.get('id', '')
                if FILE_ID.fullmatch(image_id) and (self.root / d['stored_name']).is_file():
                    self.index[image_id] = d
            except (ValueError, OSError, KeyError):
                pass

    def add(self, filename: str, content: bytes):
        safe = clean_name(filename)
        if not content or len(content) > MAX_FILE_BYTES:
            raise ValueError('图片大小必须在 1 字节至 32 MiB 之间')
        try:
            with Image.open(io.BytesIO(content)) as im:
                if im.format not in ('PNG', 'JPEG', 'WEBP'):
                    raise ValueError('图片编码与允许的类型不符')
                width, height = im.size
                if width * height > MAX_PIXELS or width < 1 or height < 1:
                    raise ValueError('图片尺寸超出设备限制')
                im.verify()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            raise ValueError('图片无法解码或格式无效') from e
        uid = uuid.uuid4().hex
        actual_suffix = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}[im.format]
        stored_name = uid + actual_suffix
        dest = self.root / stored_name
        dest.write_bytes(content)
        record = {
            'id': uid, 'display_name': safe, 'stored_name': stored_name,
            'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content),
            'width': width, 'height': height, 'uploaded_at': now(),
        }
        try:
            store_json(self.root / (uid + '.json'), record)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        with self.lock:
            self.index[uid] = record
        return dict(record)

    def get(self, uid: str):
        if not FILE_ID.fullmatch(str(uid)):
            raise KeyError('Invalid upload id')
        with self.lock:
            d = self.index.get(uid)
        if d is None:
            raise KeyError('Unknown upload id')
        return dict(d)

    def path(self, uid: str):
        return self.root / self.get(uid)['stored_name']


class JobManager:
    def __init__(self, data_root: Path, project_root: Path, store: UploadStore, start_worker=True):
        self.root = data_root / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self.store = store
        self.lock = threading.RLock()
        self.jobs = {}
        self.q = queue.Queue(maxsize=32)
        for p in self.root.glob('*/job.json'):
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                jid = d.get('id', '')
                if not FILE_ID.fullmatch(jid):
                    continue
                if d.get('status') in ('queued', 'running'):
                    d['status'] = 'interrupted'
                    d['error'] = '服务重启导致任务中断；可重新创建任务'
                    store_json(p, d)
                self.jobs[jid] = d
            except (OSError, ValueError):
                pass
        if start_worker:
            self.worker = threading.Thread(target=self._loop, daemon=True, name='single-rknn-job-worker')
            self.worker.start()

    def create(self, arm: str, image_ids: list[str]):
        if arm not in VALID_ARM:
            raise ValueError('模型名称无效')
        if not isinstance(image_ids, list) or not 1 <= len(image_ids) <= MAX_JOB_IMAGES:
            raise ValueError('每个任务必须提供 1～200 张图片')
        if any(not isinstance(uid, str) or not FILE_ID.fullmatch(uid) for uid in image_ids):
            raise ValueError('无效的图片 ID')
        if len(image_ids) != len(set(image_ids)):
            raise ValueError('图片 ID 不可重复')
        meta = [self.store.get(uid) for uid in image_ids]
        jid = uuid.uuid4().hex
        folder = self.root / jid
        folder.mkdir(parents=True, exist_ok=False)
        d = {
            'id': jid, 'arm': arm, 'image_ids': image_ids,
            'input_files': [m['display_name'] for m in meta],
            'status': 'queued', 'done': 0, 'total': len(meta),
            'created_at': now(), 'started_at': None, 'finished_at': None,
            'error': None, 'summary': None,
        }
        with self.lock:
            self.jobs[jid] = d
            store_json(folder / 'job.json', d)
            try:
                self.q.put_nowait(jid)
            except queue.Full:
                self.jobs.pop(jid, None)
                shutil.rmtree(folder)
                raise ValueError('任务队列已满，请稍后再试')
        return self.get(jid)

    def get(self, jid: str):
        if not FILE_ID.fullmatch(str(jid)):
            raise KeyError('Invalid job id')
        with self.lock:
            if jid not in self.jobs:
                raise KeyError('Unknown job id')
            return dict(self.jobs[jid])

    def recent(self):
        with self.lock:
            jobs = sorted(self.jobs.values(), key=lambda x: x['created_at'], reverse=True)[:30]
            return [dict(j) for j in jobs]

    def _patch(self, jid, **kwargs):
        with self.lock:
            self.jobs[jid].update(kwargs)
            store_json(self.root / jid / 'job.json', self.jobs[jid])

    def _loop(self):
        while True:
            jid = self.q.get()
            try:
                self._run(jid)
            except Exception as exc:
                self._patch(jid, status='failed', error=f'{type(exc).__name__}: {exc}', finished_at=now())
            finally:
                self.q.task_done()

    def _run(self, jid: str):
        d = self.get(jid)
        arm = d['arm']
        folder = self.root / jid
        inputs = folder / 'input'
        inputs.mkdir(exist_ok=False)
        engine = self.project_root / 'scripts/early9_async_pipeline_v1/run_async.py'
        if not engine.is_file():
            raise FileNotFoundError(f'原有异步推理程序不存在: {engine}')
        model_dir = self.project_root / 'models' / EXPECTED_MODELS[arm]
        if len(list(model_dir.glob('*.rknn'))) != 1:
            raise FileNotFoundError(f'模型文件缺失或数量异常: {model_dir}')
        self._patch(jid, status='running', started_at=now())
        for idx, uid in enumerate(d['image_ids']):
            item = self.store.get(uid)
            suffix = Path(item['stored_name']).suffix
            shutil.copy2(self.store.path(uid), inputs / f'{idx:04d}_{uid}{suffix}')
        out = folder / 'output'
        cmd = [sys.executable, '-u', str(engine),
               '--arm', arm, '--project-root', str(self.project_root),
               '--input', str(inputs), '--out-dir', str(out),
               '--core', '0_1_2', '--warmup', '10', '--queue-depth', '3',
               '--save-visual', 'comparison', '--progress-every', '10']
        (folder / 'command.json').write_text(json.dumps(cmd, ensure_ascii=False, indent=2), encoding='utf-8')
        with (folder / 'engine.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(self.project_root))
            progress_prev = -1
            while proc.poll() is None:
                try:
                    if (out / 'predictions.jsonl').is_file():
                        raw = (out / 'predictions.jsonl').read_bytes()
                        count = raw.count(b'\n')
                        if count != progress_prev:
                            progress_prev = count
                            self._patch(jid, done=min(d['total'], count))
                except OSError:
                    pass
                time.sleep(.4)
            code = proc.wait()
        if code:
            tail = (folder / 'engine.log').read_text(encoding='utf-8', errors='replace')[-2500:]
            raise RuntimeError(f'NPU 推理返回错误码 {code}；日志末尾：{tail}')
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        rows = self._read_rows(jid)
        if len(rows) != d['total'] or summary.get('n_images') != d['total']:
            raise RuntimeError('推理结果数量与提交图片数量不一致')
        for idx, (row, uid) in enumerate(zip(rows, d['image_ids'])):
            if row['index'] != idx or row['input_sha256'] != self.store.get(uid)['sha256']:
                raise RuntimeError(f'输入原图校验不一致，位置 {idx}')
        short_summary = {
            'n_images': summary['n_images'],
            'processing_fps': summary['full_pipeline_images_per_second'],
            'phase_ms': summary.get('phase_ms', {}),
            'prediction_counts': summary.get('prediction_counts', {}),
            'model_sha256': summary.get('model_sha256'),
            'measurement': '异步处理吞吐量：含解码、CPU/NPU、证据PNG写盘；不含上传、模型装载与浏览器显示',
        }
        self._patch(jid, status='completed', done=d['total'], finished_at=now(), summary=short_summary)

    def _read_rows(self, jid):
        path = self.root / jid / 'output/predictions.jsonl'
        if not path.is_file():
            return []
        return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

    def results(self, jid):
        job = self.get(jid)
        if job['status'] != 'completed':
            return []
        rows = self._read_rows(jid)
        output = []
        for i, row in enumerate(rows):
            uid = job['image_ids'][i]
            record = {
                'index': i, 'upload_id': uid, 'filename': job['input_files'][i],
                'prediction': row['prediction'], 'logit': row['logit'],
                'support_score': row['probability_fake'],
                'top10_patch_indices': row['top10_patch_indices'],
                'top10_patch_rc': row['top10_patch_rc'],
                'top10_region_counts': row['top10_region_counts'],
                'timings_ms': row['timings_ms'],
                'image_url': f'/api/uploads/{uid}/raw',
                'visual_url': f'/api/jobs/{jid}/visual/{Path(row["visual_file"]).name}' if row.get('visual_file') else None,
            }
            output.append(record)
        return output

    def zip_path(self, jid):
        job = self.get(jid)
        if job['status'] != 'completed':
            raise ValueError('任务尚未完成')
        folder = self.root / jid
        archive = folder / 'report_export.zip'
        if archive.is_file():
            return archive
        tmp = folder / 'report_export.tmp'
        with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as z:
            for rel in ('job.json', 'output/summary.json', 'output/predictions.jsonl'):
                z.write(folder / rel, arcname=rel)
            z.writestr('readme.txt', 'EviRoute Edge 离线检测结果。Top-10 为 Fake-support patch，不是经标注的伪影区域。\n')
            for p in sorted((folder / 'output/visuals').glob('*.png')):
                z.write(p, arcname='visuals/' + p.name)
        os.replace(tmp, archive)
        return archive


class OfflineHandler(BaseHTTPRequestHandler):
    server_version = 'EviRouteEdge/' + VERSION
    def log_message(self, fmt, *args):
        print('%s %s' % (self.log_date_time_string(), fmt % args), flush=True)

    def _headers(self, code, ctype, length, cache='no-store', extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(length))
        self.send_header('Cache-Control', cache)
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        if extra:
            for name, value in extra.items():
                self.send_header(name, value)
        self.end_headers()

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode('utf-8')
        self._headers(code, 'application/json; charset=utf-8', len(b))
        self.wfile.write(b)

    def _file(self, path: Path, mime: str, download_name=None):
        size = path.stat().st_size
        extra = {'Content-Disposition': f'attachment; filename="{download_name}"'} if download_name else None
        self._headers(200, mime, size, extra=extra)
        with path.open('rb') as stream:
            shutil.copyfileobj(stream, self.wfile)

    def _body(self, max_bytes):
        try:
            n = int(self.headers.get('Content-Length', '-1'))
        except ValueError:
            raise ValueError('无效的 Content-Length')
        if n < 1 or n > max_bytes:
            raise ValueError('请求内容超出允许大小')
        data = self.rfile.read(n)
        if len(data) != n:
            raise ValueError('请求内容不完整')
        return data

    def _dispatch(self, method):
        path = unquote(urlsplit(self.path).path)
        svc = self.server.service
        if method == 'GET':
            if path == '/' or path in ('/app.js', '/style.css'):
                name = 'index.html' if path == '/' else path[1:]
                self._file(STATIC / name, {'index.html':'text/html; charset=utf-8', 'app.js':'text/javascript; charset=utf-8', 'style.css':'text/css; charset=utf-8'}[name])
                return
            if path == '/api/status':
                self._json(200, {'version':VERSION, 'offline':True, 'core':'0_1_2',
                                 'arms':{a: len(list((svc.project_root/'models'/EXPECTED_MODELS[a]).glob('*.rknn'))) == 1 for a in VALID_ARM}})
                return
            if path == '/api/jobs':
                self._json(200, {'jobs':svc.jobs.recent()})
                return
            match = re.fullmatch(r'/api/uploads/([0-9a-f]{32})/raw', path)
            if match:
                p = svc.uploads.path(match.group(1))
                self._file(p, MIME[p.suffix.lower()])
                return
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})', path)
            if match:
                self._json(200, svc.jobs.get(match.group(1)))
                return
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/results', path)
            if match:
                self._json(200, {'results':svc.jobs.results(match.group(1))})
                return
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/export', path)
            if match:
                self._file(svc.jobs.zip_path(match.group(1)), 'application/zip', download_name='eviroute_' + match.group(1)[:8] + '.zip')
                return
            match = re.fullmatch(r'/api/jobs/([0-9a-f]{32})/visual/([a-zA-Z0-9_-]+\.png)', path)
            if match:
                svc.jobs.get(match.group(1))
                p = svc.jobs.root/match.group(1)/'output/visuals'/match.group(2)
                if not p.is_file():
                    raise KeyError('Image missing')
                self._file(p, 'image/png')
                return
            raise KeyError('页面或资源不存在')
        if method == 'POST':
            if path == '/api/uploads':
                name = unquote(self.headers.get('X-File-Name', 'image.png'))
                raw = self._body(MAX_FILE_BYTES)
                self._json(201, svc.uploads.add(name, raw))
                return
            if path == '/api/jobs':
                payload = json.loads(self._body(64 * 1024).decode('utf-8'))
                if not isinstance(payload, dict):
                    raise ValueError('JSON 必须是一个对象')
                job = svc.jobs.create(payload.get('arm'), payload.get('image_ids'))
                self._json(202, job)
                return
            raise KeyError('API 不存在')
        raise ValueError('不支持的 HTTP 方法')

    def do_GET(self):
        try:
            self._dispatch('GET')
        except (ValueError, KeyError) as e:
            self._json(400 if isinstance(e, ValueError) else 404, {'error':str(e)})
        except (OSError, RuntimeError) as e:
            self._json(500, {'error':str(e)})

    def do_POST(self):
        try:
            self._dispatch('POST')
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._json(400 if not isinstance(e, KeyError) else 404, {'error':str(e)})
        except (OSError, RuntimeError) as e:
            self._json(500, {'error':str(e)})


class OfflineService:
    def __init__(self, project_root: Path, data_root: Path, start_worker=True):
        self.project_root = Path(project_root).resolve()
        self.uploads = UploadStore(Path(data_root).resolve()/'uploads')
        self.jobs = JobManager(Path(data_root).resolve(), self.project_root, self.uploads, start_worker=start_worker)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path('/root/EviRoute_RK3588'))
    p.add_argument('--data-root', type=Path, help='Persistent local job/upload directory, default project/eviroute_edge_data')
    p.add_argument('--host', default='127.0.0.1', help='Default loopback-only. 0.0.0.0 needs network isolation/authentication.')
    p.add_argument('--port', type=int, default=8765)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = args.data_root or args.project_root/'eviroute_edge_data'
    service = OfflineService(args.project_root, data)
    if args.host not in ('127.0.0.1', '::1', 'localhost'):
        print('WARNING: no access authentication. Exposing to LAN is NOT recommended in V1.', flush=True)
    server = ThreadingHTTPServer((args.host, args.port), OfflineHandler)
    server.service = service
    print(f'EVIRoute Edge OFFLINE UI READY http://{args.host}:{args.port} (device-local; no Internet needed)', flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
