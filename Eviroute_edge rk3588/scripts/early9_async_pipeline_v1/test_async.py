import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_async as app

class FakeLite:
    NPU_CORE_0_1_2 = 'mock'
    def __init__(self,verbose=False):pass
    def load_rknn(self,path):return 0
    def init_runtime(self,core_mask):return 0
    def inference(self,inputs,data_format):
        assert inputs[0].shape == (1,768,1,196)
        assert inputs[0].dtype == np.float32
        assert data_format == ['nchw']
        a = inputs[0].reshape(1,3,16,16,14,14).mean(axis=(1,2,3)).reshape(-1)
        return [a + np.arange(196,dtype=np.float32)*np.float32(0.0001)]
    def release(self):pass

class TestAsync(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.model=self.root/'mock.rknn';self.model.write_bytes(b'mock')
        self.inp=self.root/'inputs';self.inp.mkdir()
        rng=np.random.default_rng(1357)
        for i in range(8):
            img=rng.integers(0,256,(59+i*3,103+i*2,3),dtype=np.uint8)
            Image.fromarray(img).save(self.inp/f'{i:03}.png' if i%2 else self.inp/f'{i:03}.jpg')
    def args(self,out,mode):
        return Namespace(arm='R_BCE',input=self.inp,project_root=self.root,
            model=self.model,out_dir=out,core='0_1_2',warmup=2,max_images=0,non_recursive=False,
            save_visual=mode,store_all_scores=True,respect_exif=False,
            queue_depth=2,progress_every=10,allow_other_arch=True)
    def test_same_as_serial(self):
        from run_images import execute as old_execute
        h=hashlib.sha256(b'mock').hexdigest()
        for mode in ('none','comparison'):
            old_args=self.args(self.root/f'{mode}_serial',mode)
            new_args=self.args(self.root/f'{mode}_async',mode)
            with patch.dict(app.old.MODEL_SHA256,{'R_BCE':h}):
                s1=old_execute(old_args,FakeLite)
                s2=app.execute(new_args,FakeLite)
            self.assertEqual(s1['n_images'],8)
            self.assertEqual(s2['n_images'],8)
            oldrows=[json.loads(x) for x in (old_args.out_dir/'predictions.jsonl').read_text().splitlines()]
            newrows=[json.loads(x) for x in (new_args.out_dir/'predictions.jsonl').read_text().splitlines()]
            for r1,r2 in zip(oldrows,newrows):
                for key in ('relative_path','input_sha256','logit','top10_patch_indices',
                            'top10_patch_scores','prediction','visual_file'):
                    self.assertEqual(r1[key],r2[key],(key,mode))
            if mode=='comparison':
                for r in oldrows:
                    rel=r['visual_file']
                    self.assertEqual(hashlib.sha256((old_args.out_dir/rel).read_bytes()).hexdigest(),
                                     hashlib.sha256((new_args.out_dir/rel).read_bytes()).hexdigest())
    def test_reject_missing_model(self):
        args=self.args(self.root/'out','none')
        args.model=self.root/'not.rknn'
        with self.assertRaises(FileNotFoundError):app.execute(args,FakeLite)
    def test_reject_wrong_sha(self):
        args=self.args(self.root/'out','none')
        with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'):app.execute(args,FakeLite)

if __name__=='__main__':unittest.main()
