#!/usr/bin/env python3
"""AutoDL CPU-only numerical + classification + evidence audit of 480-image RK3588 outputs."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from common import (ARMS,DOMAINS,MODEL_SHA256,read_meta_inputs,verify_sha,sha256,top10,region_counts,json_save)


def auc(y,s):
    if len(set(y))!=2:raise ValueError('AUC requires both classes')
    return float(roc_auc_score(y,s))


def metrics_for(y, pt, board):
    return {'n':int(len(y)),'n_real':int(np.count_nonzero(y==0)),'n_fake':int(np.count_nonzero(y==1)),
            'auc_pytorch':auc(y,pt),'auc_board':auc(y,board),'auc_delta_board_minus_pytorch':auc(y,board)-auc(y,pt),
            'ap_pytorch':float(average_precision_score(y,pt)),
            'ap_board':float(average_precision_score(y,board)),
            'acc_pytorch_at_zero':float(np.mean((pt>=0)==y)),
            'acc_board_at_zero':float(np.mean((board>=0)==y))}



def bootstrap_shared_real(y, dom, pt, board, domains, draws, seed):
    """Same 240 REAL resampled ONCE per bootstrap replicate for ALL source AUCs.

    Each source supplies 30 independent Fake images. Repeated real comparisons
    are correlated and must not be resampled separately for each source.
    """
    rng=np.random.default_rng(seed)
    real=np.flatnonzero(y==0)
    if real.size!=240:raise RuntimeError('Expected exactly 240 unique shared real images')
    fake={d:np.flatnonzero((dom==d)&(y==1)) for d in domains}
    if any(x.size!=30 for x in fake.values()):raise RuntimeError('Expected 30 fake images for every generator')
    out=np.empty(draws,dtype=np.float64)
    for j in range(draws):
        sr=rng.choice(real,real.size,replace=True)
        dif=[]
        for d in domains:
            sf=rng.choice(fake[d],fake[d].size,replace=True)
            idx=np.concatenate([sr,sf]); dif.append(auc(y[idx],board[idx])-auc(y[idx],pt[idx]))
        out[j]=float(np.mean(dif))
    real_idx=np.flatnonzero(y==0)
    delta=[]
    for d in domains:
        idx=np.concatenate([real_idx,fake[d]])
        delta.append(auc(y[idx],board[idx])-auc(y[idx],pt[idx]))
    return {'point':float(np.mean(delta)),
            'ci95_low':float(np.percentile(out,2.5)),
            'ci95_high':float(np.percentile(out,97.5)),
            'bootstrap_n':draws,'seed':seed,
            'method':'shared 240 Real resampled ONCE per draw for ALL domains, each 30 Fake resampled per source; within-image paired PT/NPU'}


def array_checks(ref, got):
    if ref.shape!=got.shape or ref.shape[1]!=196:raise ValueError('bad input score dimensions')
    a=ref.astype(np.float64);b=got.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('non-finite scores')
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    if not np.all(den>0):raise RuntimeError('zero-norm score vector')
    cs=(a*b).sum(axis=1)/den
    errors=np.abs(a-b)
    return cs,errors.mean(axis=1),errors.max(axis=1)


def bootstrap_macros(y,dom, pt,board, domains, draws,seed):
    rng=np.random.default_rng(seed)
    grouped=[]
    for domain in domains:
        fi=np.flatnonzero((dom==domain)&(y==1)); ri=np.flatnonzero((dom==domain)&(y==0))
        if len(fi)!=30 or len(ri)!=30:raise RuntimeError('Bad domain balance in bootstrap')
        grouped.append((ri,fi))
    out=np.empty(draws,dtype=np.float64)
    for rep in range(draws):
        deltas=[]
        for ri,fi in grouped:
            si=np.concatenate([rng.choice(ri,len(ri),replace=True),rng.choice(fi,len(fi),replace=True)])
            deltas.append(auc(y[si],board[si])-auc(y[si],pt[si]))
        out[rep]=np.mean(deltas)
    return {'point':float(np.mean([metrics_for(y[dom==d],pt[dom==d],board[dom==d])['auc_delta_board_minus_pytorch'] for d in domains])),
            'ci95_low':float(np.percentile(out,2.5)),'ci95_high':float(np.percentile(out,97.5)),
            'bootstrap_n':draws,'seed':seed,'method':'paired by image, stratified within each fixed generator and label; generator set fixed'}


def per_image_rows(meta,z,board):
    ids=z['global_index'];labels=z['labels'];domains=z['domain']
    output=[];metrics_by_arm={}
    for arm in ARMS:
        s_ref=z[arm+'_scores'];s_obs=board[arm]
        cosine,mae,mx=array_checks(s_ref,s_obs)
        pt_z=np.empty(len(ids),dtype=np.float64);npu_z=np.empty(len(ids),dtype=np.float64)
        overlaps=np.empty(len(ids),dtype=np.int64)
        for j in range(len(ids)):
            a,aidx=top10(s_ref[j]);b,bidx=top10(s_obs[j]);pt_z[j]=a;npu_z[j]=b
            overlaps[j]=len(set(aidx.tolist())&set(bidx.tolist()))
            ref_reg=region_counts(aidx); board_reg=region_counts(bidx)
            output.append({'global_index':int(ids[j]),'domain':str(domains[j]),'label':int(labels[j]),'arm':arm,
                           'pytorch_logit':a,'board_logit':b,'abs_logit_error':abs(a-b),
                           'prediction_match':int((a>=0)==(b>=0)),
                           'pytorch_pred':int(a>=0),'board_pred':int(b>=0),'pytorch_margin_abs':abs(a),
                           'cosine':float(cosine[j]),'mae':float(mae[j]),'max_abs':float(mx[j]),
                           'top10_overlap':int(overlaps[j]),'pytorch_top10_b1_count':ref_reg[0],
                           'board_top10_b1_count':board_reg[0],
                           'pytorch_top10_b2only_count':ref_reg[1],
                           'board_top10_b2only_count':board_reg[1]})
        metrics_by_arm[arm]={'z_pytorch':pt_z,'z_board':npu_z,'cosine':cosine,'mae':mae,'max_abs':mx,'overlap':overlaps}
    return output,metrics_by_arm


def analyze(args):
    meta,z,ins=read_meta_inputs(args.data_dir)
    checks=json.loads((args.data_dir/'transport_sha256.json').read_text(encoding='utf-8'))
    for f in ('references.npz','inputs_224_uint8.npz','manifest.json'):
        verify_sha(args.data_dir/f,checks[f],f)
    scores={}
    board_checks={}
    for arm in ARMS:
        path=args.board_results_dir/f'{arm}_full480_board_outputs.npz'
        reportpath=args.board_results_dir/f'{arm}_full480_board_report.json'
        report=json.loads(reportpath.read_text(encoding='utf-8'))
        if report.get('status')!='completed' or report.get('n')!=480 or report.get('arm')!=arm:
            raise RuntimeError('Board report not full480 or wrong model '+arm)
        verify_sha(path,report['board_outputs_sha256'],f'{arm} board output')
        if report['model_sha256']!=MODEL_SHA256[arm] or report['reference_npz_sha256']!=checks['references.npz']:
            raise RuntimeError('Board source/model identity mismatch: '+arm)
        with np.load(path,allow_pickle=False) as q:
            if not np.array_equal(q['global_index'],z['global_index']):
                raise RuntimeError('Board sample order mismatch: '+arm)
            if str(q['model_sha256'].item())!=MODEL_SHA256[arm] or str(q['ref_scores_sha256'].item())!=checks['references.npz'] or str(q['ref_input_sha256'].item())!=checks['inputs_224_uint8.npz']:
                raise RuntimeError('Board NPZ identity mismatch: '+arm)
            obs=q['patch_scores'].copy()
            if obs.shape!=(480,196) or obs.dtype!=np.float32 or not np.isfinite(obs).all():
                raise RuntimeError('Bad board score array: '+arm)
            scores[arm]=obs
        board_checks[arm]={'board_sha256':sha256(path),'model_sha256':MODEL_SHA256[arm]}
    rows,m=per_image_rows(meta,z,scores)
    dom=z['domain'];y=z['labels'].astype(np.int64)
    report={'status':'completed',
        'protocol':'SAME 480 previously inspected Detail4 images, 30 unique Real+30 Fake per source; shared real uniquely partitioned across domains',
        'caution':['EXPLORATORY Eval8000; not an independent final holdout.',
            'Balanced-60 AUCs use an artificial non-overlapping 30-real partition per source; shared-real-270 AUC uses 240 unique real samples reused ANALYTICALLY and bootstraps them jointly; neither equals the original Eval8000 500-real/500-fake protocol.',
            'Device numerical agreement does NOT prove board causal faithfulness; check perturbations separately if claimed.',
            'All metrics are based on these specific two fixed checkpoints, 480 images and this RKNN runtime/hardware.'],
        'provenance':{'references_sha256':checks['references.npz'],
                      'inputs_sha256':checks['inputs_224_uint8.npz'],
                      'board_outputs':board_checks,'detail4_cache_sha256':meta['provenance']['detail4_patch_cache_sha256']},
        'arms':{},'comparison':{}}
    for arm in ARMS:
        x=m[arm];zpt=x['z_pytorch'];zb=x['z_board'];overlap=x['overlap']
        group={}
        for d in DOMAINS:
            mask=dom==d
            group[d]=metrics_for(y[mask],zpt[mask],zb[mask])
        shared_real=np.flatnonzero(y==0)
        shared_270={}
        for d in DOMAINS:
            idx=np.concatenate([shared_real,np.flatnonzero((dom==d)&(y==1))])
            shared_270[d]=metrics_for(y[idx],zpt[idx],zb[idx])
        seen=[d for d in DOMAINS if d!='SD1.4']
        macro7=float(np.mean([group[d]['auc_delta_board_minus_pytorch'] for d in seen]))
        obs_fake=(y==1)
        fake_entries=[r for r in rows if r['arm']==arm and r['label']==1]
        report['arms'][arm]={
            'pooled_480':metrics_for(y,zpt,zb), 'per_source_balanced_60':group,
            'per_source_shared_real_270':shared_270,
            'unseen7_macro_shared_real_270_auc_pytorch':float(np.mean([shared_270[d]['auc_pytorch'] for d in seen])),
            'unseen7_macro_shared_real_270_auc_board':float(np.mean([shared_270[d]['auc_board'] for d in seen])),
            'unseen7_macro_shared_real_270_auc_delta_ci95':bootstrap_shared_real(y,dom,zpt,zb,seen,args.bootstrap,args.seed+1),
            'unseen7_macro_auc_pytorch':float(np.mean([group[d]['auc_pytorch'] for d in seen])),
            'unseen7_macro_auc_board':float(np.mean([group[d]['auc_board'] for d in seen])),
            'unseen7_macro_auc_delta_board_minus_pytorch_ci95':bootstrap_macros(y,dom,zpt,zb,seen,args.bootstrap,args.seed),
            'numerical':{'min_cosine':float(x['cosine'].min()),'mean_cosine':float(x['cosine'].mean()),
                'mean_mae':float(x['mae'].mean()),'max_abs_error':float(x['max_abs'].max()),
                'mean_top10_overlap':float(overlap.mean()),
                'top10_overlap_hist':{str(j):int(np.sum(overlap==j)) for j in range(11) if np.any(overlap==j)},
                'top10_exact_count':int(np.count_nonzero(overlap==10)),
                'prediction_agreement_count':int(np.count_nonzero((zpt>=0)==(zb>=0))),
                'prediction_agreement_rate':float(np.mean((zpt>=0)==(zb>=0))),
                'max_abs_logit_error':float(np.max(np.abs(zpt-zb))),
                'mean_abs_logit_error':float(np.mean(np.abs(zpt-zb))),
                'margin_within_1_count':int(np.sum(np.abs(zpt)<1)),
                'margin_within_1_prediction_flips':int(np.sum((np.abs(zpt)<1)&((zpt>=0)!=(zb>=0))))},
            'fake_spatial':{
                'reference_top10_B1':float(np.mean([r['pytorch_top10_b1_count']/10 for r in fake_entries])),
                'board_top10_B1':float(np.mean([r['board_top10_b1_count']/10 for r in fake_entries])),
                'reference_top10_B2_only':float(np.mean([r['pytorch_top10_b2only_count']/10 for r in fake_entries])),
                'board_top10_B2_only':float(np.mean([r['board_top10_b2only_count']/10 for r in fake_entries])),
            },
        }
    for domain in DOMAINS:
        report['comparison'][domain]={a:{
            'auc_pytorch':report['arms'][a]['per_source_balanced_60'][domain]['auc_pytorch'],
            'auc_board':report['arms'][a]['per_source_balanced_60'][domain]['auc_board']} for a in ARMS}
    out=args.out.resolve()
    if out.exists():raise FileExistsError('Refusing overwrite of analysis folder: '+str(out))
    out.mkdir(parents=True)
    with (out/'per_image_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'per_source_comparison.csv').open('w',encoding='utf-8',newline='') as f:
        fieldnames=['arm','domain','n','auc_pytorch','auc_board','auc_delta_board_minus_pytorch','ap_pytorch','ap_board','acc_pytorch_at_zero','acc_board_at_zero']
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
        for arm in ARMS:
            for domain in DOMAINS:
                d=report['arms'][arm]['per_source_balanced_60'][domain]
                w.writerow({'arm':arm,'domain':domain,**{k:d[k] for k in fieldnames if k not in ('arm','domain')}})
    json_save(out/'summary.json',report)
    print('\nAUTO-DL BOARD 480 ANALYSIS: PASS')
    for arm in ARMS:
        x=report['arms'][arm]
        print(arm, 'prediction=', x['numerical']['prediction_agreement_count'],'/480',
              'top10_avg=',round(x['numerical']['mean_top10_overlap'],4),
              'unseen7 AUC PT=',round(100*x['unseen7_macro_auc_pytorch'],3),
              'NPU=',round(100*x['unseen7_macro_auc_board'],3),
              'AUCdelta CI95=',x['unseen7_macro_auc_delta_board_minus_pytorch_ci95'],
              'sharedReal240 macro AUC PT=',round(100*x['unseen7_macro_shared_real_270_auc_pytorch'],3),
              'NPU=',round(100*x['unseen7_macro_shared_real_270_auc_board'],3),flush=True)
    print('Saved:',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--board-results-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261010)
    args=p.parse_args()
    if not 100<=args.bootstrap<=20000:p.error('--bootstrap must be 100..20000')
    analyze(args)

if __name__=='__main__':main()
