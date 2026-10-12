#!/usr/bin/env python3
"""Offline comparison of two complete JPEG/PNG pipeline runs on identical image files.

No claims about correctness unless an independent labels CSV is explicitly supplied.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load_run(folder: Path, arm: str):
    folder=Path(folder)
    summ=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if summ['status']!='complete' or summ['arm']!=arm:
        raise RuntimeError(f'Invalid completed {arm} run: {folder}')
    rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows)!=summ['n_images']:
        raise RuntimeError(f'{arm} incomplete per-image predictions')
    if len(set(x['relative_path'] for x in rows))!=len(rows):
        raise RuntimeError(f'{arm} duplicate relative paths')
    return summ,rows


def load_labels(file, expected_paths):
    labels={}
    with Path(file).open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        if not {'relative_path','label'}<=set(reader.fieldnames or []):
            raise ValueError('Label CSV must have columns relative_path,label')
        for r in reader:
            key=r['relative_path']
            if key in labels or r['label'] not in ('0','1'):
                raise ValueError(f'Duplicate or invalid label for {key}')
            labels[key]=int(r['label'])
    if set(labels)!=set(expected_paths):
        raise ValueError(f'Label keys mismatch: {len(set(expected_paths)-set(labels))} missing, {len(set(labels)-set(expected_paths))} extra')
    return labels


def evaluate(args):
    bsum,b=load_run(Path(args.rbce_run),'R_BCE')
    dsum,d=load_run(Path(args.rdebias_run),'R_Debias')
    if len(b)!=len(d) or any(x['relative_path']!=y['relative_path'] or not x.get('input_sha256') or x['input_sha256']!=y.get('input_sha256') for x,y in zip(b,d)):
        raise RuntimeError('Paired runs must use EXACTLY the same image paths, order and input SHA256')
    if bsum['protocol']['decode']!=dsum['protocol']['decode']:
        raise RuntimeError('Preprocessing mismatch between arms')
    if bsum['core']!=dsum['core'] or bsum['protocol']['save_visual']!=dsum['protocol']['save_visual']:
        raise RuntimeError('Core mask or visualization mode differs, not comparable')
    records=[]
    for x,y in zip(b,d):
        xx=set(x['top10_patch_indices']); yy=set(y['top10_patch_indices'])
        records.append({
            'relative_path': x['relative_path'], 'input_sha256': x.get('input_sha256'),
            'R_BCE_pred': x['prediction'], 'R_Debias_pred': y['prediction'],
            'R_BCE_logit': x['logit'], 'R_Debias_logit': y['logit'],
            'R_BCE_prob_fake': x['probability_fake'], 'R_Debias_prob_fake': y['probability_fake'],
            'pred_agreement': int(x['prediction']==y['prediction']),
            'inter_model_top10_overlap': len(xx&yy),
            'R_BCE_B1': x['top10_region_counts']['b1_count'],
            'R_Debias_B1': y['top10_region_counts']['b1_count'],
            'R_BCE_full_image_ms': x['timings_ms']['full_image_ms'],
            'R_Debias_full_image_ms': y['timings_ms']['full_image_ms'],
        })
    summ={
        'n':len(records),
        'warning': 'Inter-model Top10 overlap is not NPU-vs-PyTorch numerical stability. No ground-truth artifact masks.',
        'inter_model_prediction_agreement':sum(x['pred_agreement'] for x in records),
        'inter_model_mean_top10_overlap':float(np.mean([x['inter_model_top10_overlap'] for x in records])),
        'R_BCE':{'images_per_second':bsum['full_pipeline_images_per_second'],
                 'full_image_ms_p95':bsum['phase_ms']['full_image_ms']['p95'],
                 'prediction_counts':bsum['prediction_counts']},
        'R_Debias':{'images_per_second':dsum['full_pipeline_images_per_second'],
                    'full_image_ms_p95':dsum['phase_ms']['full_image_ms']['p95'],
                    'prediction_counts':dsum['prediction_counts']},
    }
    if args.labels_csv:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        labels=load_labels(args.labels_csv,[x['relative_path'] for x in records])
        truth=np.asarray([labels[x['relative_path']] for x in records])
        if len(np.unique(truth))!=2:
            raise ValueError('AUC needs both Real and Fake labels')
        for arm, rows in [('R_BCE',b),('R_Debias',d)]:
            scores=np.array([x['logit'] for x in rows],dtype=np.float64)
            summ[arm]['labeled_metrics']={
                'auc':float(roc_auc_score(truth,scores)),
                'ap':float(average_precision_score(truth,scores)),
                'acc_at_zero':float(accuracy_score(truth,(scores>=0).astype(int))),
                'n_real':int(np.sum(truth==0)), 'n_fake':int(np.sum(truth==1))
            }
    out=Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing overwrite: {out}')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'paired_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0].keys()))
        writer.writeheader();writer.writerows(records)
    (out/'comparison_summary.json').write_text(json.dumps(summ,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('IMAGE PIPELINE PAIR COMPARISON: PASS')
    print(json.dumps(summ,indent=2,ensure_ascii=False))
    return summ


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rbce-run',type=Path,required=True)
    p.add_argument('--rdebias-run',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--labels-csv',type=Path,help='Optional independent labels CSV with relative_path,label columns')
    evaluate(p.parse_args())


if __name__=='__main__':main()
