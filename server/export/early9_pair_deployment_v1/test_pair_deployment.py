#!/usr/bin/env python3
"""CPU-only unit checks; no checkpoints, GPU, RKNN installation required."""
import unittest
import numpy as np
from export.early9_pair_deployment_v1.pair_common import (
    pack_rgb_to_nchw,topk_logit,stats,summarize,strict_float_gate,
    frozen_top10_audit,CHECKPOINTS
)

class TestPairDeployment(unittest.TestCase):
    def test_packing_order_and_contiguous(self):
        x=np.arange(3*224*224,dtype=np.float32).reshape(1,3,224,224)
        p=pack_rgb_to_nchw(x)
        self.assertEqual(p.shape,(1,768,1,196))
        self.assertTrue(p.flags.c_contiguous)
        for ch in (0,1,2):
            for r,c in ((0,0),(2,4),(13,13)):
                token=r*14+c
                for u,v in ((0,0),(3,8),(15,15)):
                    self.assertEqual(p[0,ch*256+u*16+v,0,token], x[0,ch,r*16+u,c*16+v])
    def test_packing_rejects_bad(self):
        with self.assertRaises(ValueError):pack_rgb_to_nchw(np.zeros((1,3,224,224),dtype=np.float16))
    def test_stats_identity(self):
        x=np.linspace(-3,4,196,dtype=np.float32)
        s=stats(x,x)
        self.assertGreater(s['cosine'],.999999)
        self.assertEqual(s['top10_overlap'],10)
        self.assertEqual(s['logit_error'],0)
        strict_float_gate(summarize([s]),'identity')
    def test_logit_topk(self):
        x=np.arange(196,dtype=np.float32)
        self.assertAlmostEqual(topk_logit(x),190.5)
    def test_exact_indices(self):
        s=np.arange(196,dtype=np.float32)
        idx=np.arange(195,185,-1)
        r=frozen_top10_audit(float(s[idx].mean()),idx,s,float(s[idx].mean()),idx[::-1])
        self.assertTrue(r['top10_set_match']);self.assertFalse(r['top10_order_match'])
    def test_near_tie_edge(self):
        s=np.zeros(196,dtype=np.float32)
        ids=np.arange(10)
        s[ids[:9]]=np.arange(10,1,-1)
        s[9]=1.0;s[10]=1.0001
        new=np.r_[ids[:9],10]
        r=frozen_top10_audit(4.0,new,s,4.0,ids)
        self.assertTrue(r['near_tie'])
    def test_far_tie_rejected(self):
        s=np.zeros(196,dtype=np.float32);ids=np.arange(10);new=np.r_[ids[:9],10]
        s[9]=1;s[10]=2
        with self.assertRaises(RuntimeError): frozen_top10_audit(4,new,s,4,ids)
    def test_bad_logit_rejected(self):
        s=np.arange(196,dtype=np.float32);ids=np.arange(195,185,-1)
        with self.assertRaises(RuntimeError): frozen_top10_audit(5,ids,s,6,ids)
    def test_known_shas(self):
        self.assertEqual(set(CHECKPOINTS),{'R_BCE','R_Debias'})
        for v in CHECKPOINTS.values():self.assertEqual(len(v[1]),64)
    def test_bad_gate(self):
        s=stats(np.arange(196,dtype=np.float32),np.arange(196,dtype=np.float32))
        s['top10_overlap']=5
        with self.assertRaises(RuntimeError):strict_float_gate(summarize([s]),'failed')

if __name__=='__main__': unittest.main()
