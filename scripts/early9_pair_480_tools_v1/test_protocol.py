#!/usr/bin/env python3
"""CPU-only local regression tests; no model or board needed."""
import tempfile
import unittest
import json
from pathlib import Path
import numpy as np
from common import pack_rgb, rgb_uint8_to_packed, top10, region_counts, read_meta_inputs, json_save
from prepare_autodl import make_cohort
from analyze_autodl import metrics_for, bootstrap_macros, bootstrap_shared_real, per_image_rows
from run_rk3588 import compare_arrays, select_indices
from common import DOMAINS

class Tests(unittest.TestCase):
    def test_patchify_exact(self):
        rng=np.random.default_rng(2)
        u=rng.integers(0,256,(224,224,3),dtype=np.uint8)
        v=np.ascontiguousarray((u.astype(np.float32)/255).transpose(2,0,1)[None])
        self.assertTrue(np.array_equal(pack_rgb(v),rgb_uint8_to_packed(u)))
        self.assertEqual(pack_rgb(v).shape,(1,768,1,196))
    def test_strict_shape(self):
        with self.assertRaises(ValueError): pack_rgb(np.zeros((1,3,224,224),dtype=np.uint8))
        with self.assertRaises(ValueError): rgb_uint8_to_packed(np.zeros((224,224,3),dtype=np.float32))
    def test_top10(self):
        z,ix=top10(np.arange(196,dtype=np.float32))
        self.assertEqual(z,190.5)
        self.assertEqual(ix[0],195)
        self.assertEqual(len(set(ix.tolist())),10)
    def test_region_counts(self):
        self.assertEqual(region_counts([0,1,2,3,4,5,6,7,8,9]),(10,0,0))
        self.assertEqual(sum(region_counts([14,15,16,17,18,19,20,21,22,23])),10)
    def test_make_cohort(self):
        r=[]
        for d in DOMAINS:
            for i in range(30):r.append({'global_index':str(len(r)),'label':'1','domain':d,'path':f'fake_{d}_{i}'})
        for i in range(240):r.append({'global_index':str(len(r)),'label':'0','domain':'ADM','path':f'real_{i}'})
        gids=np.arange(480,dtype=np.int64)
        labels=np.array([1]*240+[0]*240,dtype=np.int8)
        a=make_cohort(r,gids,labels,real_partition_seed=20261010)
        b=make_cohort(r,gids,labels,real_partition_seed=20261010)
        self.assertEqual(a,b)
        for dom in DOMAINS:
            self.assertEqual(sum(t['domain']==dom and t['label']==0 for t in a),30)
            self.assertEqual(sum(t['domain']==dom and t['label']==1 for t in a),30)
        self.assertEqual(len(set(t['path'] for t in a)),480)
    def test_bad_alignment(self):
        r=[{'global_index':str(i),'label':'1' if i<240 else '0','domain':DOMAINS[min(i//30,7)],'path':str(i)} for i in range(480)]
        with self.assertRaises(RuntimeError):make_cohort(r,np.arange(480)[::-1],np.array([1]*240+[0]*240),real_partition_seed=1)
    def test_rknn_comparison_identical(self):
        s=np.arange(196,dtype=np.float32)+1
        a=compare_arrays(s,s)
        self.assertAlmostEqual(a['cosine'],1)
        self.assertEqual(a['top10_overlap'],10)
        self.assertEqual(a['logit_abs_error'],0)
    def test_auc_bootstrap_paired_no_change(self):
        y=np.array([0]*240+[1]*240)
        dom=np.array([d for d in DOMAINS for _ in range(30)]*2)
        x=np.where(y==1,1.,-1.)+np.random.default_rng(0).normal(0,.5,480)
        full=metrics_for(y,x,x)
        self.assertEqual(full['auc_delta_board_minus_pytorch'],0)
        b=bootstrap_macros(y,dom,x,x,list(DOMAINS),100,42)
        self.assertEqual(b['point'],0)
        self.assertEqual(b['ci95_low'],0)
        self.assertEqual(b['ci95_high'],0)
    def test_auc_bootstrap_shared_real_no_change(self):
        y=np.array([0]*240+[1]*240)
        dom=np.array([d for d in DOMAINS for _ in range(30)]*2)
        x=np.where(y==1,1.,-1.)+np.random.default_rng(0).normal(0,.5,480)
        b=bootstrap_shared_real(y,dom,x,x,list(DOMAINS),100,42)
        self.assertEqual(b['point'],0)
        self.assertEqual(b['ci95_low'],0)
        self.assertEqual(b['ci95_high'],0)
    def test_per_image_rows(self):
        s=np.tile(np.linspace(-1,1,196,dtype=np.float32),(2,1))
        fake={'global_index':np.array([3,4]),'labels':np.array([1,0]),'domain':np.array(['ADM','ADM'])}
        for arm in ('R_BCE','R_Debias'):fake[arm+'_scores']=s
        r,m=per_image_rows({},fake,{'R_BCE':s,'R_Debias':s})
        self.assertEqual(len(r),4)
        self.assertEqual(m['R_BCE']['overlap'].tolist(),[10,10])
    def test_smoke_balanced(self):
        y=np.array([1]*240+[0]*240,dtype=np.int8)
        dom=np.array([d for d in DOMAINS for _ in range(30)]*2)
        ids=select_indices(y,dom,16)
        self.assertEqual(len(ids),16)
        for d in DOMAINS:
            self.assertEqual(sum(dom[i]==d and y[i]==0 for i in ids),1)
            self.assertEqual(sum(dom[i]==d and y[i]==1 for i in ids),1)
    def test_cache_load_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            gids=np.arange(480,dtype=np.int64); labels=np.array([1]*240+[0]*240,dtype=np.int8)
            dom=np.array([d for d in DOMAINS for _ in range(30)]*2)
            score=np.tile(np.arange(196,dtype=np.float32),(480,1))
            ix=np.argsort(-score,axis=1)[:,:10].astype(np.int16)
            z=score[np.arange(480)[:,None],ix].mean(axis=1)
            np.savez_compressed(p/'references.npz',global_index=gids,labels=labels,domain=dom,
                                R_BCE_scores=score,R_BCE_logits=z,R_BCE_top10=ix,
                                R_Debias_scores=score,R_Debias_logits=z,R_Debias_top10=ix)
            np.savez_compressed(p/'inputs_224_uint8.npz',global_index=gids,
                                rgb=np.zeros((480,224,224,3),dtype=np.uint8))
            json_save(p/'manifest.json',{'global_indices':gids.tolist(),'labels':labels.tolist(),
                                        'domains':dom.tolist()})
            meta,ref,ins=read_meta_inputs(p)
            self.assertEqual(len(ins['rgb']),480)
            self.assertEqual(len(ref['R_BCE_scores']),480)
            # Test strict sample order gate.
            json_save(p/'manifest.json',{'global_indices':gids[::-1].tolist(),'labels':labels.tolist(),'domains':dom.tolist()})
            with self.assertRaises(RuntimeError):read_meta_inputs(p)

if __name__=='__main__':unittest.main(verbosity=2)
