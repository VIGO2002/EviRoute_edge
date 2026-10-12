#!/usr/bin/env python3
"""Offline tests: never load models or initiate hardware inference."""
import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from PIL import Image

import server


def jpeg_bytes():
    b=io.BytesIO()
    Image.new('RGB',(18,14),(30,70,130)).save(b,format='JPEG')
    return b.getvalue()


class TestOfflineUI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.project=self.root/'project'
        self.project.mkdir()
        self.img=jpeg_bytes()
        self.store=server.UploadStore(self.root/'data/uploads')

    def test_clean_name_path_sanitization(self):
        self.assertEqual(server.clean_name('../../image.JPG'),'image.jpg')
        self.assertEqual(server.clean_name('folder\\a.png'),'a.png')
        with self.assertRaises(ValueError):server.clean_name('evil.sh')

    def test_valid_upload_and_hash(self):
        record=self.store.add('abc.jpeg',self.img)
        self.assertTrue(server.FILE_ID.fullmatch(record['id']))
        self.assertEqual(record['sha256'],hashlib.sha256(self.img).hexdigest())
        self.assertEqual(record['width'],18)
        self.assertEqual(self.store.path(record['id']).read_bytes(),self.img)

    def test_upload_persistent(self):
        record=self.store.add('abc.jpeg',self.img)
        reopened=server.UploadStore(self.root/'data/uploads')
        self.assertEqual(reopened.get(record['id'])['sha256'],record['sha256'])

    def test_reject_invalid_image(self):
        with self.assertRaises(ValueError):self.store.add('x.jpg',b'not-an-image')
        with self.assertRaises(ValueError):self.store.add('x.exe',self.img)

    def test_job_validation(self):
        store=self.store
        manager=server.JobManager(self.root/'data',self.project,store,start_worker=False)
        uid=store.add('x.jpg',self.img)['id']
        with self.assertRaises(ValueError):manager.create('wrong',[uid])
        with self.assertRaises(ValueError):manager.create('R_BCE',[uid,uid])
        with self.assertRaises(ValueError):manager.create('R_BCE',[])
        with self.assertRaises(ValueError):manager.create('R_BCE',['bad-id'])

    def test_persist_jobs_without_restarting_interrupted(self):
        uid=self.store.add('x.jpg',self.img)['id']
        manager=server.JobManager(self.root/'data',self.project,self.store,start_worker=False)
        job=manager.create('R_BCE',[uid])
        self.assertEqual(job['status'],'queued')
        manager2=server.JobManager(self.root/'data',self.project,self.store,start_worker=False)
        self.assertEqual(manager2.get(job['id'])['status'],'interrupted')

    def test_http_static_and_upload_and_submit(self):
        svc=server.OfflineService(self.project,self.root/'data',start_worker=False)
        srv=ThreadingHTTPServer(('127.0.0.1',0),server.OfflineHandler)
        srv.service=svc
        t=threading.Thread(target=srv.serve_forever,daemon=True)
        t.start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        base=f'http://127.0.0.1:{srv.server_port}'
        with urllib.request.urlopen(base+'/') as resp:
            self.assertIn(b'EviRoute',resp.read())
            self.assertIn('127.0.0.1',json.dumps({'host':'127.0.0.1'}))
        req=urllib.request.Request(base+'/api/uploads',data=self.img,headers={'X-File-Name':'test.jpg'},method='POST')
        with urllib.request.urlopen(req) as resp:
            uid=json.load(resp)['id']
        with urllib.request.urlopen(base+f'/api/uploads/{uid}/raw') as resp:
            self.assertEqual(resp.read(),self.img)
        body=json.dumps({'arm':'R_BCE','image_ids':[uid]}).encode()
        req=urllib.request.Request(base+'/api/jobs',data=body,headers={'Content-Type':'application/json'},method='POST')
        with urllib.request.urlopen(req) as resp:
            job=json.load(resp)
            self.assertEqual(resp.status,202)
        with urllib.request.urlopen(base+'/api/jobs/'+job['id']) as resp:
            self.assertEqual(json.load(resp)['status'],'queued')
        with urllib.request.urlopen(base+'/api/jobs') as resp:
            self.assertEqual(len(json.load(resp)['jobs']),1)

    def test_full_job_with_mock_engine(self):
        script=self.project/'scripts/early9_async_pipeline_v1/run_async.py'
        script.parent.mkdir(parents=True)
        script.write_text('''import argparse,hashlib,json,time\nfrom pathlib import Path\nfrom PIL import Image\np=argparse.ArgumentParser()\nfor f in ["arm","project-root","input","out-dir","core","warmup","queue-depth","save-visual","progress-every"]:p.add_argument("--"+f)\na=p.parse_args();out=Path(a.out_dir);out.mkdir(parents=True);(out/"visuals").mkdir()\nfiles=sorted(Path(a.input).glob("*"));rows=[]\nfor i,x in enumerate(files):\n im=Image.open(x); im.resize((448,224)).save(out/"visuals"/(str(i)+".png"))\n rows.append(dict(index=i,input_sha256=hashlib.sha256(x.read_bytes()).hexdigest(),prediction="FAKE",logit=2.0,probability_fake=.88,top10_patch_indices=list(range(10)),top10_patch_rc=[[0,j] for j in range(10)],top10_region_counts=dict(b1_count=10),timings_ms=dict(full_image_ms=100),visual_file="visuals/"+str(i)+".png"))\n(out/"predictions.jsonl").write_text("".join(json.dumps(r)+"\\n" for r in rows))\n(out/"summary.json").write_text(json.dumps(dict(n_images=len(rows),full_pipeline_images_per_second=11.2,prediction_counts=dict(REAL=0,FAKE=len(rows)),model_sha256="mock",phase_ms={})))\n''',encoding='utf-8')
        model=self.project/'models/early9_rbce_fp16/a.rknn'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'unit-test-only-not-a-real-model')
        svc=server.OfflineService(self.project,self.root/'data',start_worker=True)
        uid=svc.uploads.add('test.jpg',self.img)['id']
        job=svc.jobs.create('R_BCE',[uid]); jid=job['id']
        stop=time.time()+8
        while time.time()<stop and svc.jobs.get(jid)['status'] in ('queued','running'):
            time.sleep(.08)
        j=svc.jobs.get(jid)
        self.assertEqual(j['status'],'completed',j.get('error'))
        results=svc.jobs.results(jid)
        self.assertEqual(len(results),1)
        self.assertEqual(results[0]['prediction'],'FAKE')
        self.assertEqual(results[0]['top10_patch_indices'],list(range(10)))
        archive=svc.jobs.zip_path(jid)
        self.assertTrue(archive.is_file())


if __name__=='__main__':unittest.main()
