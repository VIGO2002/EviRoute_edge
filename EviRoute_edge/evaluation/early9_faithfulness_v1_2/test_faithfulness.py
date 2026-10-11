import sys
import unittest
import tempfile
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_early9_rgb_faithfulness as f

class FaithfulnessUnitTests(unittest.TestCase):
    def test_rings(self):
        self.assertEqual(Counter(f.zone(i) for i in range(196)),Counter({0:52,1:44,2:100}))

    def test_matched_controls_exclude_selected(self):
        import random
        selected=[0,1,14,15,16,29,30,31,60,160]
        for seed in range(50):
            rnd=f.matching_control(selected,random.Random(seed))
            self.assertEqual(len(set(rnd)),10)
            self.assertFalse(set(rnd)&set(selected))
            self.assertEqual(Counter(map(f.zone,rnd)),Counter(map(f.zone,selected)))

    def test_local_mean_same_as_original(self):
        import importlib.util
        old=Path(__file__).resolve().parent/'reference_make_causal_perturbations_clean_v2.py'
        spec=importlib.util.spec_from_file_location('original_mean_test',old)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rng=np.random.default_rng(8)
        a=rng.integers(0,256,size=(224,224,3),dtype=np.uint8)
        im=Image.fromarray(a,'RGB')
        selected=[0,7,16,65,160,195]
        expected=np.array(module.perturb_image_224(im,selected,perturb='local_mean'))
        actual=np.array(f.local_mean_image(im,selected))
        np.testing.assert_array_equal(actual,expected)

    def test_selected_index_audit(self):
        gids=np.array([30,40]);cache=np.array([1.,2.]);idx=np.tile(np.arange(10,dtype=np.int64),(2,1))
        scores=np.zeros((2,196),dtype=np.float64)
        f.audit_clean((cache,scores,idx),cache.copy(),idx.copy(),'test',gids)
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL NUMERICAL AUDIT FAIL'):
            f.audit_clean((cache+.1,scores,idx),cache,idx,'test',gids)
        # Same ten locations in different rank order is permissible for the Top-10 mask.
        detail=f.audit_clean((cache,scores,np.roll(idx,1,axis=1)),cache,idx,'test',gids)
        self.assertTrue(detail['top10_set_exact_match'])
        self.assertEqual(detail['top10_rank_order_changed_count'],2)
        self.assertEqual(detail['top1_index_changed_count'],2)
        # Genuine selected-set changes must still abort.
        replaced=idx.copy();replaced[1,9]=42
        with self.assertRaisesRegex(RuntimeError,'changed_top10_membership=1'):
            f.audit_clean((cache,scores,replaced),cache,idx,'test',gids)
        # Tiny clean-output discrepancy remains permissible, larger discrepancy cannot pass.
        with self.assertRaisesRegex(RuntimeError,'bad_logits=1'):
            f.audit_clean((cache+np.array([0.,.5]),scores,idx),cache,idx,'test',gids)

    def test_near_tie_strict_requires_explicit_opt_in(self):
        gid=np.array([7952]);original=np.arange(10,dtype=np.int64)[None,:]
        new=original.copy();new[0,9]=79
        scores=np.full((1,196),-100.,dtype=np.float32)
        scores[0,0:9]=np.arange(20.,11.,-1.)
        scores[0,79]=11.;scores[0,9]=10.99998
        z=np.array([17.],dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError,'explicitly_allowed=0'):
            f.audit_clean((z,scores,new),z,original,'R_BCE',gid)
        r=f.audit_clean((z,scores,new),z,original,'R_BCE',gid,max_near_ties=1)
        self.assertEqual(r['allowed_near_tie_count'],1)
        self.assertEqual(r['top10_membership_changed_global_indices'],[7952])
        self.assertFalse(r['top10_set_exact_match'])
        self.assertEqual(r['allowed_near_tie_details'][0]['old_only'],[9])
        self.assertEqual(r['allowed_near_tie_details'][0]['new_only'],[79])

    def test_near_tie_rejects_large_gap_or_top9_changes(self):
        gid=np.array([7952]);original=np.arange(10,dtype=np.int64)[None,:]
        new=original.copy();new[0,9]=79
        scores=np.full((1,196),-100.,dtype=np.float32)
        scores[0,0:9]=np.arange(20.,11.,-1.)
        scores[0,79]=11.;scores[0,9]=10.1
        z=np.array([17.],dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL NUMERICAL AUDIT FAIL'):
            f.audit_clean((z,scores,new),z,original,'R_BCE',gid,max_near_ties=2)
        # If high-ranked indices changed, never permit even a near-tie.
        scores[0,9]=10.99998
        new2=new.copy();new2[0,8]=90
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL NUMERICAL AUDIT FAIL'):
            f.audit_clean((z,scores,new2),z,original,'R_BCE',gid,max_near_ties=2)

    def test_statistic(self):
        rows=[{'advantage':2.,'win':1,'d_sel':5.,'mean_d_ctrl':3.,'fake_correct':1},
              {'advantage':-2.,'win':0,'d_sel':-1.,'mean_d_ctrl':1.,'fake_correct':0}]
        s=f.mean_stat(rows)
        self.assertEqual(s['n'],2)
        self.assertEqual(s['advantage'],0)
        self.assertEqual(s['win_rate'],.5)

    def test_mock_full_input_perturbation_integration(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td)
            mock=d/'mock_eval_early9_p0.py'
            mock.write_text("import numpy as np\nfrom PIL import Image\nP0_CKPT=None\nclass MockModel:\n    def eval(self): return self\ndef load_model(device):\n    return MockModel(),None,{'paired_p0_20k':{'arm':P0_CKPT.parent.name.replace('_','-'),'step':2625}}\ndef inference(model,cls,paths,device,batch,workers):\n    zz=[]\n    for p in paths:\n        with Image.open(p) as im:\n            im=im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)\n            v=float(np.asarray(im,dtype=np.float64).mean()/255.)\n        zz.append(v)\n    scores=np.zeros((len(paths),196),dtype=np.float32)\n    top=np.tile(np.arange(10,dtype=np.int64),(len(paths),1))\n    return np.asarray(zz,dtype=np.float64),scores,top\n",encoding='utf-8')
            rows=[]
            for i in range(2):
                arr=np.random.default_rng(i).integers(0,256,size=(224,224,3),dtype=np.uint8)
                fp=d/f'image_{i}.png';Image.fromarray(arr).save(fp)
                rows.append({'path':str(fp),'global_index':i,'domain':'ADM'})
            mod=f.source_module(mock,'test_original_mock_1')
            orig=mod.inference(None,None,[r['path'] for r in rows],None,2,0)
            cache={'R_BCE_logits':np.asarray(orig[0]),'R_BCE_top10_indices':np.asarray(orig[2])}
            ck=d/'R_BCE'/'final_step.pt';ck.parent.mkdir()
            audit,results=f.evaluate_arm(d,mock,ck,'R_BCE',rows,cache,None,2,0,2,3,42)
            self.assertEqual(audit['n'],2)
            self.assertEqual(len(results),2)
            self.assertTrue(all(np.isfinite(r['advantage']) for r in results))

    def test_bootstrap_shapes(self):
        rows=[]
        for arm,_ in f.ARM_SPECS:
            for d in f.DOMAIN_ORDER:
                for i in range(5):
                    rows.append({'global_index':f.DOMAIN_ORDER.index(d)*100+i,'arm':arm,'domain':d,'advantage':i/10,'win':int(i>0)})
        b=f.bootstrap_macro(rows,replicates=100,seed=9)
        self.assertIn('unseen7_R_BCE_advantage',b['estimates'])
        self.assertIn('unseen7_R_Debias_minus_R_BCE_win',b['estimates'])
        self.assertEqual(len(b['estimates']),12)

if __name__=='__main__':unittest.main(verbosity=2)
