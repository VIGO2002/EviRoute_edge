import sys
import tempfile
import unittest
import json
from pathlib import Path
from argparse import Namespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import compare_runs as cr

class Tests(unittest.TestCase):
    def test_compare_equal_paths(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for arm in ['R_BCE','R_Debias']:
                folder=root/arm; folder.mkdir()
                (folder/'summary.json').write_text(json.dumps({
                    'status':'complete','arm':arm,'n_images':2,'core':'0_1_2',
                    'protocol':{'decode':'Pillow','save_visual':'none'},'full_pipeline_images_per_second':8,
                    'phase_ms':{'full_image_ms':{'p95':121}},'prediction_counts':{'REAL':1,'FAKE':1}}))
                with (folder/'predictions.jsonl').open('w') as f:
                    for idx,path in enumerate(['real.jpg','fake.png']):
                        f.write(json.dumps({'relative_path':path,'prediction':['REAL','FAKE'][idx],
                            'logit':[-1,1][idx],'input_sha256':'abc123','probability_fake':[.26,.73][idx],
                            'top10_patch_indices':list(range(10)),
                            'top10_region_counts':{'b1_count':10},
                            'timings_ms':{'full_image_ms':100}})+'\n')
            args=Namespace(rbce_run=root/'R_BCE',rdebias_run=root/'R_Debias',out_dir=root/'out',labels_csv=None)
            result=cr.evaluate(args)
            self.assertEqual(result['inter_model_prediction_agreement'],2)
            self.assertEqual(result['inter_model_mean_top10_overlap'],10)
            with self.assertRaises(FileExistsError):cr.evaluate(args)
    def test_labels_csv_validation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'labels.csv'
            p.write_text('relative_path,label\na.jpg,0\nb.jpg,1\n')
            self.assertEqual(cr.load_labels(p,['a.jpg','b.jpg']),{'a.jpg':0,'b.jpg':1})
            with self.assertRaises(ValueError): cr.load_labels(p,['c.jpg'])

if __name__=='__main__':unittest.main()
