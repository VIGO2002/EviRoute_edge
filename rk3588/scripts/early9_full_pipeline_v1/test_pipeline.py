#!/usr/bin/env python3
"""CPU tests only: preprocessing equivalence and real-file pipeline with a mock NPU."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_images as app

class FakeLite:
    NPU_CORE_0_1_2 = 'npu3'
    def __init__(self, verbose=False): self.calls = 0
    def load_rknn(self, path): return 0
    def init_runtime(self, core_mask): return 0
    def inference(self, inputs, data_format):
        assert data_format == ['nchw']
        assert inputs[0].shape == (1,768,1,196)
        assert inputs[0].dtype == np.float32
        self.calls += 1
        return [np.linspace(-1, 1, 196, dtype=np.float32)[None, :]]
    def release(self): pass

class Tests(unittest.TestCase):
    def test_preprocessing(self):
        rng=np.random.default_rng(88)
        x=rng.integers(0,256,(224,224,3),dtype=np.uint8)
        y=app.rgb_uint8_to_packed(x)
        ref=np.ascontiguousarray((x.astype(np.float32)/np.float32(255)).transpose(2,0,1)[None])
        expected=ref.reshape(1,3,14,16,14,16).transpose(0,1,3,5,2,4).reshape(1,768,1,196)
        self.assertTrue(np.array_equal(y,expected))
    def test_extreme_sigmoid(self):
        self.assertEqual(app.sigmoid(1000),1)
        self.assertEqual(app.sigmoid(-1000),0)
        self.assertAlmostEqual(app.sigmoid(0),.5)
    def test_top10_and_mask(self):
        v=np.arange(196,dtype=np.float32)
        z,ids=app.logit_and_top10(v)
        self.assertEqual(ids.tolist(),list(range(195,185,-1)))
        self.assertAlmostEqual(z,np.mean(np.arange(186,196)))
        self.assertEqual(sum(app.evidence_stats(ids).values()),10)
    def test_visual_geometry(self):
        x=np.zeros((224,224,3),dtype=np.uint8)
        ids=np.arange(10)
        self.assertEqual(app.create_visual(x,ids,'overlay').size,(224,224))
        self.assertEqual(app.create_visual(x,ids,'comparison').size,(448,224))
    def test_input_order(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)
            for n in ('z.JPG','b.png','a.jpeg','ignore.txt'):
                (d/n).touch()
            images,_=app.collect_images(d)
            self.assertEqual([x.name for x in images],['a.jpeg','b.png','z.JPG'])
    def test_read_pillow_bicubic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'a.png'
            inp=np.random.default_rng(11).integers(0,255,(300,125,3),dtype=np.uint8)
            Image.fromarray(inp).save(p)
            got,size,digest=app.read_rgb224(p)
            self.assertEqual(size,(125,300))
            self.assertEqual(digest,hashlib.sha256(p.read_bytes()).hexdigest())
            ref=np.asarray(Image.open(p).convert('RGB').resize((224,224),app.BICUBIC))
            self.assertTrue(np.array_equal(got,ref))
    def test_model_hash_rejects_wrong(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.rknn';p.write_bytes(b'mock model')
            opts=Namespace(allow_other_arch=True,model=p,project_root=Path(t),arm='R_BCE',input=Path(t),
                           non_recursive=False,max_images=0,out_dir=Path(t)/'out')
            with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):
                app.execute(opts, FakeLite)
    def test_mock_full_file_pipeline(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); model=root/'x.rknn'; model.write_bytes(b'mock')
            model_hash=hashlib.sha256(b'mock').hexdigest()
            inp=root/'input'; inp.mkdir()
            a=np.zeros((29,73,3),dtype=np.uint8);a[:,:,1]=170
            Image.fromarray(a).save(inp/'a.png')
            Image.fromarray(a).save(inp/'b.jpg')
            out=root/'results'
            args=Namespace(allow_other_arch=True,model=model,project_root=root,arm='R_BCE',input=inp,
                           non_recursive=False,max_images=0,out_dir=out,core='0_1_2',warmup=2,
                           save_visual='comparison',store_all_scores=False,respect_exif=False,progress_every=1)
            with patch.dict(app.MODEL_SHA256,{'R_BCE':model_hash}):
                summary=app.execute(args, FakeLite)
            self.assertEqual(summary['n_images'],2)
            self.assertEqual(len(list((out/'visuals').glob('*.png'))),2)
            rows=[json.loads(x) for x in (out/'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(len(rows[0]['top10_patch_indices']),10)
            self.assertEqual(rows[0]['prediction'],'FAKE')
            self.assertEqual(rows[0]['original_size_wh'],[73,29])
            self.assertGreater(summary['full_pipeline_images_per_second'],0)
            with patch.dict(app.MODEL_SHA256,{'R_BCE':model_hash}):
                with self.assertRaisesRegex(FileExistsError,'overwrite'):
                    app.execute(args, FakeLite)

if __name__=='__main__': unittest.main()
