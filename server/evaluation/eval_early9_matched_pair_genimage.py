#!/usr/bin/env python3
"""Strict SAME-MANIFEST paired GenImage Eval8000 comparison: R-BCE vs R-Debias.

Runs fresh R-BCE inference against verified, frozen R-Debias inference from the
completed previous evaluation; verifies both checkpoints, model restoration,
input manifest SHA, image index order, and source statistics.
No training, no hyperparameter / model selection. Explored data, diagnostic only.
"""
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import early9_pair_genimage_common_v1 as common

ARM_BCE='R-BCE'
DEFAULT_ROOT=Path('/root/autodl-tmp/EviRoute_edge')
DOMAINS=common.SOURCES_GENIMAGE
UNSEEN=tuple(d for d in DOMAINS if d!='SD1.4')
HARD=common.HARD


def verify_bce_ckpt(ckpt, path, root, debias_identity):
    info=ckpt.get('paired_p0_20k')
    if not isinstance(info,dict):
        raise RuntimeError('R-BCE has no paired_p0_20k identity')
    req={'schema':common.CKPT_SCHEMA,'arm':ARM_BCE,'step':2625,
         'original_train_mode_bug_preserved':True}
    for k,v in req.items():
        if info.get(k)!=v:
            raise RuntimeError(f'R-BCE identity mismatch: {k}: {info.get(k)!r}')
    if info.get('debias_config') is not None:
        raise RuntimeError('R-BCE checkpoint must contain NO debias config')
    di=debias_identity['identity']
    if info.get('shared_init_sha256')!=di.get('shared_init_sha256'):
        raise RuntimeError('Initial checkpoint SHA differs between R-BCE and R-Debias')
    if info.get('signature')!=di.get('signature'):
        raise RuntimeError('Training source / splits / manifest differ between paired models')
    cfg=ckpt.get('lora_config')
    if not isinstance(cfg,dict) or cfg.get('blocks')!=[6,7,8] or cfg.get('targets')!=['qkv','proj']:
        raise RuntimeError('R-BCE LoRA architecture mismatch')
    if ckpt.get('best_epoch')!=3:
        raise RuntimeError('R-BCE expected Epoch 3 Step 2625')
    summary=json.loads((root/common.PAIR_DIR/'R_BCE'/'summary.json').read_text())
    if (summary.get('status')!='completed' or summary.get('final_step')!=2625 or
        summary.get('best_step')!=2625 or summary.get('signature')!=info['signature'] or
        summary.get('shared_init_sha256')!=info['shared_init_sha256']):
        raise RuntimeError('R-BCE checkpoint/summary provenance mismatch')
    exp=float(summary['final_val_metrics']['auc'])
    actual=float(ckpt['best_val_metrics']['auc'])
    if abs(exp-actual)>1e-9:
        raise RuntimeError('R-BCE checkpoint / summary validation AUC mismatch')
    return {'arm':ARM_BCE,'step':2625,'checkpoint_path':str(path),
            'checkpoint_sha256':common.sha256(path),
            'shared_init_sha256':info['shared_init_sha256'],
            'expected_val_auc':exp}


def load_verified_debias_cache(root, df, identity):
    directory=root/'evaluation/early9_RDebias_genimage8000_v1'
    summary_path=directory/'summary.json'
    npz_path=directory/'candidate_outputs.npz'
    if not summary_path.is_file() or not npz_path.is_file():
        raise FileNotFoundError('Run existing R-Debias GenImage8000 evaluator FIRST; missing '+str(directory))
    prior=json.loads(summary_path.read_text())
    if prior.get('dataset')!='GenImage Eval8000' or prior.get('checkpoint',{}).get('checkpoint_sha256')!=identity['checkpoint_sha256']:
        raise RuntimeError('R-Debias cached result does not match verified checkpoint')
    if prior.get('checkpoint',{}).get('identity')!=identity['identity']:
        raise RuntimeError('R-Debias cache identity differs from currently loaded checkpoint')
    with np.load(npz_path,allow_pickle=False) as array:
        expected_names=df['global_index'].astype(str).to_numpy()
        labels=df['label'].to_numpy(dtype=np.int64)
        if (not {'file_names','labels','logits','patch_scores','top10_indices'}.issubset(array.files) or
            not np.array_equal(array['file_names'],expected_names) or
            not np.array_equal(array['labels'],labels)):
            raise RuntimeError('R-Debias cache image ordering/labels do not match exact Eval8000 manifest')
        z,scores,top10=common.check_tensors(array['logits'],array['patch_scores'],array['top10_indices'])
    if len(z)!=8000:
        raise RuntimeError('Cached R-Debias does not have 8000 records')
    recomputed=common.metrics(labels,z,top10)
    for k in ('auc','ap','fake_top1_B1','fake_top10_B1','real_top10_B1'):
        if abs(float(prior['candidate'][k])-recomputed[k])>1e-7:
            raise RuntimeError(f'Cached R-Debias summary disagreeing with source scores on {k}')
    for source in DOMAINS:
        take=(df['domain']==source).to_numpy(dtype=bool)
        sm=common.metrics(labels[take],z[take],top10[take])
        prior_source=prior['per_source'][source]
        for k in ('auc','fake_top1_B1','fake_top10_B1'):
            if abs(sm[k]-float(prior_source[k]))>1e-7:
                raise RuntimeError(f'R-Debias cache source metrics differ: {source} {k}')
    return z,top10,{'summary_sha256':common.sha256(summary_path),
                    'outputs_sha256':common.sha256(npz_path),
                    'fully_rechecked_against_manifest':True}


def paired_metrics(labels,bce_z,bce_ix,deb_z,deb_ix):
    bce=common.metrics(labels,bce_z,bce_ix)
    deb=common.metrics(labels,deb_z,deb_ix)
    numeric=('auc','ap','accuracy_at_zero','fake_tpr_at_zero','real_tnr_at_zero',
             'fake_top1_B1','fake_top10_B1','fake_top1_B2','fake_top10_B2',
             'real_top1_B1','real_top10_B1')
    delta={k:deb[k]-bce[k] for k in numeric}
    agreement=float(np.mean((np.asarray(bce_z)>=0)==(np.asarray(deb_z)>=0)))
    overlap=float(np.mean([len(set(a.tolist()) & set(b.tolist()))
                          for a,b in zip(bce_ix,deb_ix)]))
    return {'R_BCE':bce,'R_Debias':deb,'delta_Debias_minus_BCE':delta,
            'predictions_agree_at_zero':agreement,'top10_overlap_mean':overlap}


def b1_ratio(indices, top1=False):
    ix=np.asarray(indices,dtype=np.int64)
    c=ix%14;r=ix//14
    b=(r==0)|(r==13)|(c==0)|(c==13)
    return float(np.mean(b[:,0] if top1 else b))


def bootstrap_paired(df,bce_z,bce_ix,deb_z,deb_ix,replicates=1000,seed=20261009):
    """Fixed-source conditional paired bootstrap; shared real 500 IDs sampled ONCE per replicate.

    Each source has 500 fake images; the 500 real files are the same across
    domains. Sampling each domain's real cohort separately would artificially
    break the dependence. Resample common real file IDs and generator-specific
    fake files, using identical draws for both models.
    """
    if replicates<100:
        raise ValueError('Need at least 100 bootstrap draws for stable diagnostics')
    labels=df['label'].to_numpy(dtype=np.int64)
    sources=df['domain'].to_numpy()
    real_base=df[(df['domain']==DOMAINS[0]) & (df['label']==0)]['path'].tolist()
    if len(real_base)!=500 or len(set(real_base))!=500:
        raise RuntimeError('Expected 500 distinct shared real files')
    by_source={}
    for domain in DOMAINS:
        take=(sources==domain)
        sub=df[take]
        real_rows=sub[sub['label']==0]
        fake_rows=sub[sub['label']==1]
        if len(real_rows)!=500 or len(fake_rows)!=500:
            raise RuntimeError(f'Unexpected class counts {domain}')
        real_map=dict(zip(real_rows['path'].tolist(),real_rows.index.tolist()))
        if set(real_map)!=set(real_base):
            raise RuntimeError('Shared 500 real paths are not identical across domains')
        by_source[domain]=(np.asarray([real_map[p] for p in real_base],dtype=np.int64),
                           fake_rows.index.to_numpy(dtype=np.int64))
    rng=np.random.default_rng(seed)
    auc_boot={domain:np.zeros(replicates,dtype=np.float64) for domain in DOMAINS}
    edge1_boot={domain:np.zeros(replicates,dtype=np.float64) for domain in DOMAINS}
    edge10_boot={domain:np.zeros(replicates,dtype=np.float64) for domain in DOMAINS}
    binary_y=np.r_[np.zeros(500,dtype=np.int64),np.ones(500,dtype=np.int64)]
    bce_z=np.asarray(bce_z,dtype=np.float64)
    deb_z=np.asarray(deb_z,dtype=np.float64)
    for rep in range(replicates):
        real_pick=rng.integers(0,500,500)
        for domain in DOMAINS:
            real_rows,fake_rows=by_source[domain]
            fake_pick=rng.integers(0,500,500)
            ix=np.concatenate([real_rows[real_pick],fake_rows[fake_pick]])
            auc_boot[domain][rep]=(roc_auc_score(binary_y,deb_z[ix])-
                                   roc_auc_score(binary_y,bce_z[ix]))
            chosen=fake_rows[fake_pick]
            edge1_boot[domain][rep]=b1_ratio(deb_ix[chosen],top1=True)-b1_ratio(bce_ix[chosen],top1=True)
            edge10_boot[domain][rep]=b1_ratio(deb_ix[chosen])-b1_ratio(bce_ix[chosen])
    def ci(samples):
        low,hi=np.percentile(samples,[2.5,97.5])
        return {'low':float(low),'high':float(hi)}
    unseen_auc=np.mean([auc_boot[d] for d in UNSEEN],axis=0)
    all_auc=np.mean([auc_boot[d] for d in DOMAINS],axis=0)
    unseen_edge1=np.mean([edge1_boot[d] for d in UNSEEN],axis=0)
    unseen_edge10=np.mean([edge10_boot[d] for d in UNSEEN],axis=0)
    return {'method':'paired stratified nonparametric bootstrap; same resampled real images across eight domains; generator-specific fake samples; fixed evaluated sources',
            'replicates':replicates,'seed':seed,
            'interpretation':'Descriptive intervals CONDITIONAL on this explored dataset and these eight fixed sources; not confidence about new unseen generators or untuned model selection.',
            'all8_macro_auc_delta_CI95':ci(all_auc),
            'unseen7_macro_auc_delta_CI95':ci(unseen_auc),
            'unseen7_fake_top1_B1_delta_CI95':ci(unseen_edge1),
            'unseen7_fake_top10_B1_delta_CI95':ci(unseen_edge10),
            'per_source_delta_auc_CI95':{d:ci(auc_boot[d]) for d in DOMAINS},
            'per_source_delta_top10_B1_CI95':{d:ci(edge10_boot[d]) for d in DOMAINS}}


def report_and_save(out,df,identity_bce,identity_deb,cached,rb_z,rb_scores,rb_ix,rd_z,rd_ix,boot):
    if out.exists():
        raise RuntimeError('Output directory exists (refusing overwrite): '+str(out))
    labels=df['label'].to_numpy(dtype=np.int64)
    sources=df['domain'].to_numpy()
    overall=paired_metrics(labels,rb_z,rb_ix,rd_z,rd_ix)
    per_source={}
    for domain in DOMAINS:
        take=sources==domain
        per_source[domain]=paired_metrics(labels[take],rb_z[take],rb_ix[take],rd_z[take],rd_ix[take])
    def agg(model_metric,key,domain_subset):
        return float(np.mean([per_source[d][model_metric][key] for d in domain_subset]))
    groups={}
    for label,names in [('all8',DOMAINS),('unseen7',UNSEEN),('difficult4',HARD)]:
        a={}
        for metric in ('auc','fake_top1_B1','fake_top10_B1','fake_top1_B2','fake_top10_B2'):
            b=agg('R_BCE',metric,names);d=agg('R_Debias',metric,names)
            a[metric]={'R_BCE':b,'R_Debias':d,'delta_Debias_minus_BCE':d-b}
        groups[label]=a
    report={
        'status':'completed','purpose':'EXPLORATORY strict SAME-MANIFEST comparison of frozen fixed-Step2625 paired training checkpoints; not an independent final held-out test',
        'dataset':'GenImage Eval8000','n_manifest_rows':8000,'n_unique_images':4500,
        'shared_real_file_count':500,'train_generator':'SD1.4','source_order':list(DOMAINS),
        'eval_manifest_sha256':identity_deb['identity']['signature']['eval_manifest_sha256'],
        'R_BCE_checkpoint':identity_bce,'R_Debias_checkpoint':identity_deb,
        'R_Debias_cached_inference_provenance':cached,
        'mode':'R_BCE freshly inferred; R_Debias previously inferred, cached outputs revalidated against identical image order, labels, checkpoint and previous metrics',
        'overall_pooled_rows':overall,'per_source':per_source,'macro':groups,
        'paired_bootstrap_CI95':boot,
        'caution':['Source-wise Macro-AUC is primary; SD1.4 excluded from unseen7.',
                   '8000 rows share the same 500 real files across eight domains; unique image count=4500.',
                   'Intervals condition on already-explored GenImage samples and fixed generators.',
                   'Edge occupancy is not evidence faithfulness or causal boundary independence.',
                   'No checkpoint choice, hyperparameter tuning, or threshold calibration is performed.']}
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    rows=[]
    for domain in DOMAINS:
        p=per_source[domain]
        rows.append({'domain':domain,'n':p['R_BCE']['n'],
                     **{f'bce_{k}':v for k,v in p['R_BCE'].items() if k!='n'},
                     **{f'debias_{k}':v for k,v in p['R_Debias'].items() if k!='n'},
                     **{f'delta_{k}':v for k,v in p['delta_Debias_minus_BCE'].items()},
                     'delta_auc_ci95_low':boot['per_source_delta_auc_CI95'][domain]['low'],
                     'delta_auc_ci95_high':boot['per_source_delta_auc_CI95'][domain]['high']})
    with (out/'per_source_paired.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'paired_outputs.npz',
        global_index=df['global_index'].to_numpy(dtype=np.int64),
        labels=labels,R_BCE_logits=np.asarray(rb_z,dtype=np.float32),
        R_Debias_logits=np.asarray(rd_z,dtype=np.float32),
        R_BCE_top10_indices=np.asarray(rb_ix,dtype=np.int16),
        R_Debias_top10_indices=np.asarray(rd_ix,dtype=np.int16))
    print('\n================ R-BCE vs R-Debias: SAME-IMAGE PAIR ================',flush=True)
    for group in ['all8','unseen7']:
        print('GROUP',group,flush=True)
        for key in ['auc','fake_top1_B1','fake_top10_B1']:
            val=groups[group][key]
            print(f'  {key:19} R-BCE={val["R_BCE"]:.6f} R-Debias={val["R_Debias"]:.6f} delta={val["delta_Debias_minus_BCE"]:+.6f}',flush=True)
    print('Unseen7 paired bootstrap AUC delta CI95:',boot['unseen7_macro_auc_delta_CI95'],flush=True)
    print('Unseen7 paired bootstrap Top10 B1 delta CI95:',boot['unseen7_fake_top10_B1_delta_CI95'],flush=True)
    print('R-BCE validation reload:',identity_bce['restored_val_auc'],flush=True)
    print('R-BCE vs R-Debias GENIMAGE PAIRED: PASS',flush=True)
    print('SAVED:',out,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--out',type=Path,default=Path('evaluation/early9_RBCE_vs_RDebias_genimage8000_v1'))
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0 or args.bootstrap<100:
        p.error('batch-size>=1, workers>=0, bootstrap>=100')
    if not torch.cuda.is_available():
        p.error('CUDA device required for R-BCE re-inference')
    root=args.root.resolve()
    out=common.absolute(root,args.out)
    if out.exists():
        raise RuntimeError('Refuse to overwrite existing output: '+str(out))
    pair=root/common.PAIR_DIR
    path_deb=pair/'R_Debias'/'final_step.pt'
    path_bce=pair/'R_BCE'/'final_step.pt'
    if not path_bce.is_file() or not path_deb.is_file():
        raise FileNotFoundError('Need both R_BCE/final_step.pt and R_Debias/final_step.pt')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    ckpt_deb=torch.load(path_deb,map_location='cpu',weights_only=False)
    identity_deb=common.check_checkpoint(ckpt_deb,path_deb,root)
    ckpt_bce=torch.load(path_bce,map_location='cpu',weights_only=False)
    identity_bce=verify_bce_ckpt(ckpt_bce,path_bce,root,identity_deb)
    df,unique_paths=common.read_genimage(root)
    manifest_sha=common.sha256(root/'evaluation/genimage_8domain_eval8000/manifest.csv')
    signature=identity_deb['identity']['signature']
    if manifest_sha!=signature.get('eval_manifest_sha256'):
        raise RuntimeError('GenImage manifest changed since the paired training provenance was recorded')
    rd_z,rd_ix,cached=load_verified_debias_cache(root,df,identity_deb)
    print('MATCHED CHECKPOINTS / GENIMAGE MANIFEST: PASS',flush=True)
    print('R-BCE SHA256:',identity_bce['checkpoint_sha256'],flush=True)
    print('R-Debias SHA256:',identity_deb['checkpoint_sha256'],flush=True)
    print('R-Debias VERIFIED CACHED OUTPUT SHA256:',cached['outputs_sha256'],flush=True)
    source=root/'evaluation/eval_early9_p0.py'
    spec=importlib.util.spec_from_file_location('paired_genimage_original_p0_eval',source)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import original Early-9 evaluator')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.P0_CKPT=path_bce
    model,cls,restored=module.load_model(torch.device('cuda:0'))
    if restored['paired_p0_20k']['arm']!=ARM_BCE or restored['paired_p0_20k']['step']!=2625:
        raise RuntimeError('Restored wrong R-BCE model')
    val_cache=root/'distillation/depth_sweep_features/vitb_early9/val.pt'
    val=torch.load(val_cache,map_location='cpu',weights_only=False,mmap=True)
    val_paths=[str(x) for x in val['paths']]
    val_y=torch.as_tensor(val['labels']).cpu().numpy().astype(np.int64)
    val_z,_,_=common.check_tensors(*module.inference(model,cls,val_paths,torch.device('cuda:0'),args.batch_size,args.workers))
    val_auc=float(roc_auc_score(val_y,val_z))
    if abs(val_auc-identity_bce['expected_val_auc'])>1e-5:
        raise RuntimeError(f'R-BCE validation AUC mismatch: saved {identity_bce["expected_val_auc"]} got {val_auc}')
    identity_bce['restored_val_auc']=val_auc
    print('R-BCE STEP2625 VALIDATION AUC RELOAD: PASS',f'{val_auc:.8f}',flush=True)
    zz,ss,ii=common.check_tensors(*module.inference(model,cls,unique_paths,torch.device('cuda:0'),args.batch_size,args.workers))
    lookup={path:i for i,path in enumerate(unique_paths)}
    take=np.asarray([lookup[p] for p in df['path']],dtype=np.int64)
    rb_z,rb_scores,rb_ix=zz[take],ss[take],ii[take]
    label=df['label'].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(np.unique(df['global_index'].to_numpy())),np.arange(8000)):
        raise RuntimeError('Manifest is not a stable 0..7999 global index')
    boot=bootstrap_paired(df,rb_z,rb_ix,rd_z,rd_ix,args.bootstrap)
    report_and_save(out,df,identity_bce,identity_deb,cached,rb_z,rb_scores,rb_ix,rd_z,rd_ix,boot)


if __name__=='__main__':
    main()
