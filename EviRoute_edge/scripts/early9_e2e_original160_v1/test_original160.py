#!/usr/bin/env python3
"""CPU synthetic fixtures only. Not a RK3588 numerical test."""
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

BASE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('prepare160',BASE/'prepare_original160_autodl.py')
pre=importlib.util.module_from_spec(s);s.loader.exec_module(pre)

class PrepTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.manifest=self.root/'source_manifest.json'
        names=pre.EXPECTED
        gid=0
        data={'status':'prepared','n_images':480,'global_indices':[],'labels':[], 'domains':[], 'source_paths':[]}
        # Tiny images are sufficient for protocol/file verification.
        for d in names:
            for label in (0,1):
                for sample in range(30):
                    ext='.png' if sample%2 else '.JPEG'
                    name=f'{gid:04d}{ext}'
                    path=self.root/'source'/name
                    path.parent.mkdir(exist_ok=True)
                    arr=np.full((16+sample,22+(sample%8),3),gid%255,dtype=np.uint8)
                    Image.fromarray(arr).save(path)
                    data['global_indices'].append(gid)
                    data['labels'].append(label)
                    data['domains'].append(d)
                    data['source_paths'].append(str(path))
                    gid+=1
        self.manifest.write_text(json.dumps(data))
        self.data=data

    def test_deterministic_strata_and_nonoverlap(self):
        x=pre.selection(self.data,10,20261010)
        y=pre.selection(self.data,10,20261010)
        self.assertEqual(x,y)
        self.assertEqual(len(x),160)
        self.assertEqual(len({r['global_index'] for r in x}),160)
        self.assertEqual(set(Counter((r['domain'],r['label']) for r in x).values()),{10})

    def test_real_original_copies_and_manifest(self):
        out,archive=pre.prepare(self.manifest,self.root/'out')
        self.assertEqual(sum(1 for p in out.rglob('*.JPEG')),0)  # stored with lowercase extension
        self.assertTrue(archive.exists())
        obj=json.loads((out/'manifest.json').read_text())
        self.assertEqual(obj['n_images'],160)
        for r in obj['images']:
            self.assertEqual(pre.sha256(out/r['relative_path']),pre.sha256(Path(r['source_path'])))
        with self.assertRaises(FileExistsError):
            pre.prepare(self.manifest,self.root/'out')

    def test_invalid_manifest_rejected(self):
        self.data['source_paths'][0]=self.data['source_paths'][1]
        with self.assertRaises(ValueError):pre.selection(self.data,10,20261010)
        with self.assertRaises(ValueError):pre.selection(self.data,5,20261010)

if __name__=='__main__':unittest.main()
