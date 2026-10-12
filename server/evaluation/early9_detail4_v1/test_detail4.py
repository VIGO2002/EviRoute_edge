#!/usr/bin/env python3
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import run_early9_detail4 as detail
import legacy_analyze_lowlevel_red_blue_v2 as legacy

class TestDetail4(unittest.TestCase):
    def setUp(self):
        self.scores=np.random.default_rng(7).normal(size=196)

    def test_fake_red_blue_exact_rings(self):
        red=np.argsort(-self.scores)[:10].tolist()
        a,b=detail.red_blue(self.scores,1,red)
        self.assertEqual(set(a),set(red))
        self.assertFalse(set(a)&set(b))
        self.assertEqual(Counter(map(detail.zone,a)),Counter(map(detail.zone,b)))
        for ring in (0,1,2):
            candidates=sorted((p for p in range(196) if detail.zone(p)==ring and p not in red), key=lambda p:(self.scores[p],p))
            self.assertEqual(set(x for x in b if detail.zone(x)==ring),set(candidates[:sum(detail.zone(x)==ring for x in red)]))

    def test_real_class_aligned(self):
        red,blue=detail.red_blue(self.scores,0)
        self.assertEqual(set(red),set(np.argsort(self.scores)[:10]))
        self.assertEqual(Counter(map(detail.zone,red)),Counter(map(detail.zone,blue)))
        self.assertTrue(np.mean(self.scores[red])<np.mean(self.scores[blue]))

    def test_fake_requires_frozen(self):
        with self.assertRaises(ValueError):detail.red_blue(self.scores,1)

    def test_all_five_and_four_metrics_legacy_parity(self):
        rgb=np.random.default_rng(4).integers(0,256,(224,224,3),dtype=np.uint8)
        image=Image.fromarray(rgb,'RGB')
        legacy_metrics=legacy.compute_patch_metrics(image)
        for k in detail.METRICS5:
            self.assertEqual(legacy_metrics[k].shape,(196,))
        red,blue=detail.red_blue(self.scores,1,np.argsort(-self.scores)[:10])
        out=detail.contrast_record(5,'ADM',1,'fake','/mock', 'R_BCE',self.scores,red,blue,legacy_metrics,0.0)
        diffs=[legacy.percentile_ranks(legacy_metrics[m])[red].mean()-legacy.percentile_ranks(legacy_metrics[m])[blue].mean() for m in detail.METRICS5]
        self.assertAlmostEqual(out['legacy5_percentile_delta'],float(np.mean(diffs)),places=12)
        self.assertAlmostEqual(out['detail4_percentile_delta'],float(np.mean([diffs[detail.METRICS5.index(k)] for k in detail.METRICS4])),places=12)

    def test_invalid_selection(self):
        with self.assertRaises(ValueError):detail.red_blue(np.ones((20,)),0)
        with self.assertRaises(ValueError):detail.red_blue(self.scores,1,[10]*10)

    def test_full_real_fake_cohort_and_nesting(self):
        shared=[f'/r/{i}' for i in range(500)]
        rows=[]
        for d in detail.DOMAINS:
            base=len(rows)
            rows.extend([{'global_index':base+i,'domain':d,'label':0,'path':p} for i,p in enumerate(shared)])
            rows.extend([{'global_index':base+500+i,'domain':d,'label':1,'path':f'/f/{d}/{i}'} for i in range(500)])
        df=pd.DataFrame(rows)
        selection=[]
        for d in detail.DOMAINS:
            selection.extend(df[(df.domain==d)&(df.label==1)].global_index.iloc[:30].tolist())
        summary={'status':'completed','n_unique_fake_images':240,'parameters':{'samples_per_source':30,'seed':20261009},'selected_global_indices':selection}
        a=detail.choose_rows(df,summary,2,16,20261009)
        b=detail.choose_rows(df,summary,30,240,20261009)
        self.assertEqual(len(a),32)
        self.assertEqual(len(b),480)
        self.assertEqual(b[b.label==0].path.nunique(),240)
        self.assertEqual(a[a.label==1].global_index.tolist(),[i for d in detail.DOMAINS for i in selection[detail.DOMAINS.index(d)*30:detail.DOMAINS.index(d)*30+2]])

    def test_bootstrap_paired_exact(self):
        rows=[]
        for domain in ('ADM','BigGAN'):
            for i in range(5):
                for arm,value in [('R_BCE',-0.2),('R_Debias',-0.1)]:
                    rows.append({'global_index':(0 if domain=='ADM' else 10)+i,'domain':domain,'label':1,'arm':arm,'detail4_percentile_delta':value})
        result=detail.macro_bootstrap(rows,['ADM','BigGAN'],['detail4_percentile_delta'],100,42)['detail4_percentile_delta']
        self.assertAlmostEqual(result['delta_Debias_minus_BCE']['point'],0.1)
        self.assertAlmostEqual(result['delta_Debias_minus_BCE']['ci95_low'],0.1)

    def test_cpu_analysis_end_to_end_mock_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'out';out.mkdir()
            (root/'evaluation/genimage_8domain_eval8000').mkdir(parents=True)
            manifest=root/'evaluation/genimage_8domain_eval8000/manifest.csv'
            manifest.write_text('mock manifest')
            src=root/'loader.py';src.write_text('mock source')
            paths=[]; drows=[]
            names=list(detail.DOMAINS)+['REAL_SHARED']*4
            rng=np.random.default_rng(15)
            for i,domain in enumerate(names):
                path=root/f'{i}.png'
                Image.fromarray(rng.integers(0,256,size=(224,224,3),dtype=np.uint8),'RGB').save(path)
                drows.append({'global_index':i,'domain':domain if domain!='REAL_SHARED' else 'ADM',
                              'label':1 if i<8 else 0,'path':str(path)})
            df=pd.DataFrame(drows)
            arrays={'global_index':np.arange(12),'label':df.label.to_numpy()}
            frozen={}
            for arm,eps in [('R_BCE',0.),('R_Debias',0.02)]:
                score=rng.normal(size=(12,196)).astype(np.float32)+eps
                arrays[arm+'_scores']=score
                arrays[arm+'_logits']=np.sort(score,axis=1)[:,-10:].mean(axis=1)
                frozen[arm+'_top10_indices']=np.argsort(-score,axis=1)[:,:10]
            np.savez_compressed(out/detail.CACHE_NAME,**arrays)
            idmap={arm:{'checkpoint_sha256':arm} for arm in detail.ARMS}
            provenance={'paired_npz_sha256':'paired','faith_summary_sha256':'faith',
                'loader_sha256':detail.sha256(src),
                'legacy_metric_source_sha256':detail.sha256(Path(detail.__file__).with_name('legacy_analyze_lowlevel_red_blue_v2.py')),
                'manifest_sha256':detail.sha256(manifest),
                'checkpoints':{arm:arm for arm in detail.ARMS}}
            (out/detail.META_NAME).write_text(json.dumps({'status':'extracted','provenance':provenance,'audits':{}}))
            fake_load=(None,src,None,df,frozen,idmap,'paired',{},'faith')
            args=SimpleNamespace(bootstrap=100,seed=20261009)
            with patch.object(detail,'load_project',return_value=fake_load):
                detail.do_analyze(args,root,out)
            self.assertTrue((out/'summary.json').is_file())
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['n_unique_images'],12)
            self.assertEqual(summary['groups']['fake_unseen7']['R_BCE']['detail4_percentile_delta']['n'],7)
            self.assertEqual(summary['groups']['real_unique']['R_Debias']['detail4_percentile_delta']['n'],4)

    def test_percentile_scale(self):
        vals=np.linspace(0,1,196)
        r=legacy.percentile_ranks(vals)
        self.assertEqual(r.min(),0.)
        self.assertEqual(r.max(),1.)

if __name__=='__main__':unittest.main(verbosity=2)
