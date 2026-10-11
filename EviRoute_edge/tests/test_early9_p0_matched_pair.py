import sys,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'distillation'))
from early9_p0_matched_pair_core import masks,masked_scores,edge_loss,compute_loss,fake_mass_stats,merge_mass,improved,CFG

class Tests(unittest.TestCase):
    def test_borders(self):
        b1,b2,b2only=masks()
        self.assertEqual(int(b1.sum()),52)
        self.assertEqual(int(b2.sum()),96)
        self.assertEqual(int(b2only.sum()),44)
        self.assertFalse(bool((b1 & b2only).any()))
    def test_loss_exact_formula(self):
        scores=torch.zeros(2,196,requires_grad=True)
        labels=torch.tensor([0.,1.])
        first=masks()[0]
        with torch.no_grad(): scores[1,first]=2.
        actual=edge_loss(scores,labels)
        p=torch.softmax(scores[1].detach(),dim=0)
        expected=.01*torch.relu(p[first].sum()-.4).square()
        self.assertTrue(torch.allclose(actual,expected))
        actual.backward()
        self.assertGreater(float(scores.grad[1].abs().sum()),0)
        self.assertEqual(float(scores.grad[0].abs().sum()),0.)
    def test_mask_logic(self):
        scores=torch.zeros(3,196)
        cfg=dict(CFG,token_dropout=0.,edge1_dropout=0.,edge2_only_dropout=0.)
        self.assertTrue(torch.equal(scores,masked_scores(scores,cfg)))
        cfg2=dict(CFG,token_dropout=0.,edge1_dropout=1.,edge2_only_dropout=0.)
        # 1.0 is allowed by original routine, and there are always unmasked interior positions.
        masked=masked_scores(scores,cfg2)
        b1,_,_=masks()
        self.assertTrue(torch.equal(masked[:,b1],torch.full((3,52),-1e4)))
        self.assertTrue(torch.equal(masked[:,~b1],torch.zeros(3,144)))
    def test_bce_branch_matches_historical(self):
        scores=torch.randn(4,196,requires_grad=True)
        logits=scores.topk(10,dim=1).values.mean(1)
        labels=torch.tensor([1.,0.,1.,0.])
        total,cls,edge=compute_loss(logits,scores,labels,'R-BCE')
        self.assertTrue(torch.equal(total,cls))
        self.assertEqual(float(edge),0.)
    def test_debias_gradients(self):
        scores=torch.randn(4,196,requires_grad=True)
        labels=torch.tensor([1.,0.,1.,0.])
        logits=scores.topk(10,dim=1).values.mean(1)
        total,_,_=compute_loss(logits,scores,labels,'R-Debias')
        total.backward()
        self.assertGreater(float(scores.grad.abs().sum()),0)
    def test_mass_stats(self):
        s=torch.ones(3,196)
        a=fake_mass_stats(s,torch.tensor([0.,1.,1.]))
        self.assertEqual(a['fake_count'],2)
        self.assertEqual(a['active_count'],0)
        result=merge_mass([a,a]); self.assertEqual(result['fake_count'],4)
    def test_historical_selection(self):
        best={'auc':.99,'bce':.1}
        self.assertTrue(improved({'auc':.991,'bce':.2},best))
        self.assertTrue(improved({'auc':.99,'bce':.09},best))
        self.assertFalse(improved({'auc':.989,'bce':.001},best))

if __name__=='__main__': unittest.main()
