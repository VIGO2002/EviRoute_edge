#!/usr/bin/env python3
"""Read-only cross-generator evaluation of P0-recipe-matched R-Debias Step 2625.

Compare against the pre-existing PyTorch P0 benchmark baselines on identical images.
This is an exploratory comparison on already-inspected cohorts, not a final held-out test.
NEVER use these cohorts to choose a model, step, training recipe, or hyperparameters.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PILOT_SHA = '2d724a6747395796dd70dba6521618516a8cc315efd63e8011e8de0c950ca548'
P0_SHA = '79f1983e83d905e0e26f2f64ec181f1800dcff80d399e31341c29c52a0a2e995'
ARM = 'R-Debias'
CKPT_SCHEMA = 'early9_matched_p0_20k_pair_v1'
EXPECTED_STEP = 2625
PAIR_DIR = 'distillation/p0_matched_pair_20k_v1'
SOURCES_PILOT = ('FLUX1-dev','DALLE-3','Imagen3','SD3','SDXL')
SOURCES_GENIMAGE = ('ADM','BigGAN','Midjourney','VQDM','GLIDE','SD1.4','SD1.5','Wukong')
HARD = ('ADM','BigGAN','VQDM','GLIDE')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def absolute(root, path):
    return path if path.is_absolute() else root/path


def check_checkpoint(ckpt, ckpt_path, root):
    """Validate identity AND pairing with R-BCE; never accept old P0 or arm C."""
    info = ckpt.get('paired_p0_20k')
    if not isinstance(info, dict):
        raise RuntimeError('Not a matched-pair checkpoint (paired_p0_20k missing)')
    for key, expected in [('schema', CKPT_SCHEMA),('arm', ARM),('step', EXPECTED_STEP),
                          ('original_train_mode_bug_preserved', True)]:
        if info.get(key) != expected:
            raise RuntimeError(f'Wrong matched-pair checkpoint {key}: {info.get(key)!r} != {expected!r}')
    cfg = info.get('debias_config')
    required = dict(token_dropout=.15,edge1_dropout=.5,edge2_only_dropout=.25,
                    edge_tau=1.,lambda_edge1=.01,rho_edge1=.4,
                    lambda_edge2=0.,rho_edge2=.6)
    if not isinstance(cfg,dict) or set(cfg)!=set(required) or any(
            abs(float(cfg[k])-v)>1e-9 for k,v in required.items()):
        raise RuntimeError(f'Wrong EviRoute debias configuration: {cfg}')
    if ckpt.get('best_epoch') != 3:
        raise RuntimeError('Expected matched P0 Step 2625 in Epoch 3')
    if not isinstance(ckpt.get('best_val_metrics'),dict) or not np.isfinite(
            float(ckpt['best_val_metrics']['auc'])):
        raise RuntimeError('Invalid stored validation metrics')
    cfg_lora=ckpt.get('lora_config',{})
    if list(cfg_lora.get('blocks',[])) != [6,7,8] or list(cfg_lora.get('targets',[])) != ['qkv','proj']:
        raise RuntimeError('Unexpected LoRA architecture')
    root_pair=root/PAIR_DIR
    shared_path=root_pair/'shared_seed42_init.pt'
    if not shared_path.is_file() or sha256(shared_path)!=info.get('shared_init_sha256'):
        raise RuntimeError('Shared seed42 initialization missing or mismatch')
    signature=info.get('signature')
    if not isinstance(signature,dict):
        raise RuntimeError('Missing source/data provenance signature')
    for key,path in [('train_manifest_sha256',root/'distillation/data_manifests/sd14_train20k_seed20261008.csv'),
                     ('base_checkpoint_sha256',root/'distillation/depth_sweep_runs/early9/mid20_edge_debiased_mil_best.pt')]:
        if not path.is_file() or sha256(path)!=signature.get(key):
            raise RuntimeError('Matched-pair source/manifest mismatch: '+str(path))
    for name,expected_sha in signature.get('source_sha256',{}).items():
        fp=root/'distillation'/name
        if not fp.is_file() or sha256(fp)!=expected_sha:
            raise RuntimeError('Historical model/train source was modified: '+str(fp))
    other_path=root_pair/'R_BCE'/'summary.json'
    debias_path=root_pair/'R_Debias'/'summary.json'
    paired_path=root_pair/'paired_summary.json'
    if not other_path.is_file() or not debias_path.is_file() or not paired_path.is_file():
        raise RuntimeError('Need both completed matched-pair summaries and paired_summary.json')
    other=json.loads(other_path.read_text())
    debias=json.loads(debias_path.read_text())
    paired=json.loads(paired_path.read_text())
    if (other.get('status')!='completed' or debias.get('status')!='completed' or
        other.get('arm')!='R-BCE' or debias.get('arm')!=ARM or
        other.get('shared_init_sha256')!=info['shared_init_sha256'] or
        debias.get('shared_init_sha256')!=info['shared_init_sha256'] or
        other.get('signature')!=signature or debias.get('signature')!=signature or
        other.get('final_step')!=EXPECTED_STEP or debias.get('final_step')!=EXPECTED_STEP or
        debias.get('best_step')!=EXPECTED_STEP or not paired.get('pairing_verified') or
        paired.get('best_validation_auc_selection',{}).get('steps',{}).get(ARM)!=EXPECTED_STEP):
        raise RuntimeError('Paired provenance/Step 2625 verification failed')
    expected_metrics=debias['final_val_metrics']
    if abs(float(expected_metrics['auc'])-float(ckpt['best_val_metrics']['auc']))>1e-8:
        raise RuntimeError('Checkpoint vs paired summary validation AUC disagrees')
    return {'identity':info, 'checkpoint_path':str(ckpt_path),
            'checkpoint_sha256':sha256(ckpt_path),
            'initialization_sha256':sha256(shared_path),
            'expected_val_auc':float(expected_metrics['auc']),
            'matched_pair_best_step':debias['best_step'],
            'matched_pair_R_BCE_val_auc':other['best_val_metrics']['auc']}


def check_tensors(logits, scores, top10):
    n = len(logits)
    logits = np.asarray(logits).reshape(-1).astype(np.float64)
    scores = np.asarray(scores).astype(np.float32)
    top10 = np.asarray(top10).astype(np.int64)
    if logits.shape != (n,) or scores.shape != (n,196) or top10.shape != (n,10):
        raise RuntimeError(f'Unexpected output shapes {logits.shape}, {scores.shape}, {top10.shape}')
    if not np.isfinite(scores).all() or not np.isfinite(logits).all():
        raise RuntimeError('Non-finite outputs')
    if np.any(top10 < 0) or np.any(top10 >= 196) or any(len(set(row)) != 10 for row in top10):
        raise RuntimeError('Duplicate / invalid Top10 Patch IDs')
    if not np.allclose(logits, np.take_along_axis(scores,top10,axis=1).mean(axis=1),rtol=1e-4,atol=1e-3):
        raise RuntimeError('Image logits do not agree with mean of Top10 Patch scores')
    return logits, scores, top10


def metrics(labels, logits, indices):
    y = np.asarray(labels, dtype=np.int64)
    z = np.asarray(logits, dtype=np.float64)
    ix = np.asarray(indices,dtype=np.int64)
    if set(np.unique(y)) != {0,1} or ix.shape != (len(y),10):
        raise RuntimeError('Metrics require both classes and Top10 indices')
    fake = y == 1
    r,c = ix//14, ix%14
    b1 = (r==0)|(r==13)|(c==0)|(c==13)
    b2 = (r<2)|(r>11)|(c<2)|(c>11)
    pred=z>=0
    return {
        'n':int(len(y)), 'n_real':int((~fake).sum()), 'n_fake':int(fake.sum()),
        'auc':float(roc_auc_score(y,z)),
        'ap':float(average_precision_score(y,z)),
        'accuracy_at_zero':float(np.mean(pred==fake)),
        'fake_tpr_at_zero':float(np.mean(pred[fake])),
        'real_tnr_at_zero':float(np.mean(~pred[~fake])),
        'fake_top1_B1':float(b1[fake,0].mean()),
        'fake_top10_B1':float(b1[fake].mean()),
        'fake_top1_B2':float(b2[fake,0].mean()),
        'fake_top10_B2':float(b2[fake].mean()),
        'real_top1_B1':float(b1[~fake,0].mean()),
        'real_top10_B1':float(b1[~fake].mean()),
    }


def read_pilot(root):
    path=root/'evaluation/aigibench_edge_pilot200/manifest.csv'
    if not path.is_file() or sha256(path) != PILOT_SHA:
        raise RuntimeError('Pilot manifest SHA256 mismatch')
    with path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows) != 200 or len({r['file_name'] for r in rows}) != 200:
        raise RuntimeError('Expected 200 unique pilot200 images')
    counts=Counter((r['source'],int(r['label'])) for r in rows)
    if set(counts)!={(s,y) for s in SOURCES_PILOT for y in (0,1)} or any(v!=20 for v in counts.values()):
        raise RuntimeError(f'Unexpected pilot200 cohort balance {counts}')
    paths=[]
    for row in rows:
        p=root/'evaluation/aigibench_edge_pilot200/images'/row['file_name']
        if not p.is_file() or sha256(p) != row['sha256']:
            raise RuntimeError('Pilot image SHA256 mismatch: '+str(p))
        if row['label_name'] != ('FAKE' if int(row['label']) else 'REAL'):
            raise RuntimeError('Pilot source label mismatch')
        paths.append(str(p))
    return rows, paths, np.asarray([int(r['label']) for r in rows],dtype=np.int64)


def read_genimage(root):
    path=root/'evaluation/genimage_8domain_eval8000/manifest.csv'
    if not path.is_file():
        raise FileNotFoundError(f'GenImage Eval8000 manifest not found: {path}')
    df=pd.read_csv(path).sort_values('global_index').reset_index(drop=True)
    needed={'global_index','domain','label','path'}
    if not needed.issubset(df.columns):
        raise RuntimeError(f'GenImage manifest missing columns {needed-set(df.columns)}')
    if len(df)!=8000 or df['global_index'].tolist()!=list(range(8000)):
        raise RuntimeError('Bad Eval8000 global indexing')
    if set(df['domain'])!=set(SOURCES_GENIMAGE):
        raise RuntimeError('Wrong GenImage source set')
    for source in SOURCES_GENIMAGE:
        part=df[df['domain']==source]
        if len(part)!=1000 or Counter(part['label'].astype(int))!={0:500,1:500}:
            raise RuntimeError('Unbalanced GenImage domain '+source)
    if df['path'].isna().any() or df['path'].duplicated().sum()!=3500:
        raise RuntimeError('Unexpected GenImage duplicate-real image composition')
    unique_paths=df['path'].drop_duplicates().tolist()
    if len(unique_paths)!=4500:
        raise RuntimeError('Expected 4500 distinct source images')
    if any(not Path(p).is_file() for p in unique_paths):
        missing=next(p for p in unique_paths if not Path(p).is_file())
        raise FileNotFoundError('Missing GenImage input file: '+missing)
    return df, unique_paths


def save_report(out, report, file_names, labels, z, scores, ids, per_source_rows, *, print_tag):
    if out.exists():
        raise RuntimeError(f'Output already exists, refusing overwrite: {out}')
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'per_source_metrics.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_source_rows[0].keys()))
        writer.writeheader();writer.writerows(per_source_rows)
    np.savez_compressed(out/'candidate_outputs.npz',file_names=np.asarray(file_names),
        labels=np.asarray(labels,dtype=np.int64),logits=np.asarray(z,dtype=np.float32),
        patch_scores=np.asarray(scores,dtype=np.float32),top10_indices=np.asarray(ids,dtype=np.int16))
    print('OVERALL:',json.dumps(report['candidate'],ensure_ascii=False),flush=True)
    if 'unseen7_macro_auc' in report:
        print('UNSEEN7_MACRO_AUC:',f"{100*report['unseen7_macro_auc']:.3f}%",flush=True)
        print('UNSEEN7_FAKE_TOP10_B1:',f"{100*report['unseen7_fake_top10_B1']:.3f}%",flush=True)
    if 'delta_vs_p0' in report:
        print('DELTA_VS_P0:',json.dumps(report['delta_vs_p0']),flush=True)
    print(print_tag+': PASS',flush=True)
    print('Saved:',out,flush=True)


def run_pilot(root, out, module, model, cls, device, args, identity):
    rows, paths, labels=read_pilot(root)
    z,scores,ids=check_tensors(*module.inference(model,cls,paths,device,args.batch_size,args.workers))
    baseline=root/'evaluation/early9_debias_baseline200/pytorch_outputs.npz'
    if not baseline.is_file():
        raise RuntimeError('P0 fixed pilot200 baseline .npz missing; cannot perform paired comparison')
    with np.load(baseline,allow_pickle=False) as old:
        names=[r['file_name'] for r in rows]
        if old['file_names'].tolist()!=names or not np.array_equal(old['labels'],labels):
            raise RuntimeError('P0 baseline does not match exact pilot200 filenames/labels')
        p0_z=old['logits'].astype(np.float64)
        p0_ids=old['top10_indices'].astype(np.int64)
    p0_metrics=metrics(labels,p0_z,p0_ids)
    cand=metrics(labels,z,ids)
    sources=[r['source'] for r in rows]
    per_source_rows=[]
    for src in SOURCES_PILOT:
        take=np.asarray([x==src for x in sources],dtype=bool)
        cm=metrics(labels[take],z[take],ids[take]);bm=metrics(labels[take],p0_z[take],p0_ids[take])
        per_source_rows.append({'source':src,**cm,'p0_auc':bm['auc'],'delta_auc':cm['auc']-bm['auc'],
                                'p0_fake_top10_B1':bm['fake_top10_B1'],
                                'delta_fake_top10_B1':cm['fake_top10_B1']-bm['fake_top10_B1']})
    overlap=[len(set(a)&set(b)) for a,b in zip(ids,p0_ids)]
    report={
        'purpose':'Exploratory SAME 200 already-inspected images; NEVER tune/select model using this cohort',
        'dataset':'AIGIBench pilot200','checkpoint':identity,
        'manifest_sha256':PILOT_SHA,'candidate':cand,'p0_pytorch':p0_metrics,
        'delta_vs_p0':{'auc':cand['auc']-p0_metrics['auc'],
                       'fake_top1_B1':cand['fake_top1_B1']-p0_metrics['fake_top1_B1'],
                       'fake_top10_B1':cand['fake_top10_B1']-p0_metrics['fake_top10_B1'],
                       'accuracy_at_zero':cand['accuracy_at_zero']-p0_metrics['accuracy_at_zero']},
        'predictions_match_p0':float(np.mean((z>=0)==(p0_z>=0))),
        'top10_overlap_p0_mean':float(np.mean(overlap)),
        'per_source':{row['source']:row for row in per_source_rows}}
    save_report(out,report,[r['file_name'] for r in rows],labels,z,scores,ids,per_source_rows,
                print_tag='EARLY9 R-DEBIAS PILOT200')


def run_genimage(root,out,module,model,cls,device,args,identity):
    df, unique_paths=read_genimage(root)
    unique_z,unique_scores,unique_ids=check_tensors(*module.inference(model,cls,unique_paths,device,args.batch_size,args.workers))
    path_idx={p:i for i,p in enumerate(unique_paths)}
    gather=np.asarray([path_idx[p] for p in df['path']],dtype=np.int64)
    z, scores, ids=unique_z[gather],unique_scores[gather],unique_ids[gather]
    labels=df['label'].to_numpy(dtype=np.int64)
    sources=df['domain'].tolist()
    per_source_rows=[]
    for src in SOURCES_GENIMAGE:
        take=(df['domain']==src).to_numpy()
        per_source_rows.append({'source':src,**metrics(labels[take],z[take],ids[take])})
    candidate=metrics(labels,z,ids)
    unseen=[r for r in per_source_rows if r['source']!='SD1.4']
    worst=min(unseen,key=lambda item:item['auc'])
    baseline_path=root/'evaluation/genimage_8domain_early9_P0/per_domain_metrics.csv'
    if not baseline_path.is_file():
        raise FileNotFoundError('Original P0 per_domain_metrics.csv required for comparison: '+str(baseline_path))
    baseline=pd.read_csv(baseline_path)
    needed={'domain','auc','fake_top1_B1','fake_top10_B1'}
    if (len(baseline)!=8 or set(baseline['domain'])!=set(SOURCES_GENIMAGE) or
            not needed.issubset(baseline.columns)):
        raise RuntimeError('P0 baseline table schema/domains mismatch')
    mapping=baseline.set_index('domain')
    p0_comparison={src:{'p0_auc':float(mapping.loc[src,'auc']),
                        'r_debias_auc':float(row['auc']),
                        'delta_auc':float(row['auc']-mapping.loc[src,'auc']),
                        'p0_fake_top1_B1':float(mapping.loc[src,'fake_top1_B1']),
                        'p0_fake_top10_B1':float(mapping.loc[src,'fake_top10_B1']),
                        'delta_fake_top1_B1':float(row['fake_top1_B1']-mapping.loc[src,'fake_top1_B1']),
                        'delta_fake_top10_B1':float(row['fake_top10_B1']-mapping.loc[src,'fake_top10_B1'])}
                   for src,row in ((r['source'],r) for r in per_source_rows)}
    # Optional stronger same-image audit based on existing historical P0 logits and top10.
    p0_npz=root/'evaluation/genimage_8domain_early9_P0/p0_outputs.npz'
    p0_same_image=None
    if p0_npz.is_file():
        with np.load(p0_npz,allow_pickle=False) as arr:
            if (set(['global_index','logits','top10_indices'])-set(arr.files) or
                    not np.array_equal(arr['global_index'],df['global_index'].to_numpy(dtype=np.int64))):
                raise RuntimeError('P0 outputs not aligned to identical GenImage eval8000 manifest')
            p0_logits=arr['logits'].astype(np.float64)
            p0_ids=arr['top10_indices'].astype(np.int64)
        if p0_logits.shape!=(8000,) or p0_ids.shape!=(8000,10):
            raise RuntimeError('Historical P0 GenImage output shape mismatch')
        for src in SOURCES_GENIMAGE:
            take=(df['domain']==src).to_numpy()
            old_metrics=metrics(labels[take],p0_logits[take],p0_ids[take])
            if abs(old_metrics['auc']-p0_comparison[src]['p0_auc'])>1e-6:
                raise RuntimeError('P0 baseline csv and logits disagree for '+src)
            if abs(old_metrics['fake_top10_B1']-p0_comparison[src]['p0_fake_top10_B1'])>1e-6:
                raise RuntimeError('P0 baseline csv and top10 disagree for '+src)
        p0_same_image={
            'p0_auc':float(roc_auc_score(labels,p0_logits)),
            'r_debias_auc':candidate['auc'],
            'delta_auc':float(candidate['auc']-roc_auc_score(labels,p0_logits)),
            'prediction_agreement_at_zero':float(np.mean((z>=0)==(p0_logits>=0))),
            'mean_top10_overlap_count':float(np.mean([len(set(a)&set(b)) for a,b in zip(ids,p0_ids)])),
            'same_manifest_order_verified':True,
        }
    # Summaries are source-weighted, not 8000 independent rows. Baseline per-domain
    # table is the pre-existing historical P0 eval, NOT produced by this run.
    p0_comparison_summary=None
    if p0_comparison:
        p0_vals=[p0_comparison[src]['p0_auc'] for src in SOURCES_GENIMAGE if src!='SD1.4']
        p0_comparison_summary={
            'p0_unseen7_macro_auc':float(np.mean(p0_vals)),
            'delta_unseen7_macro_auc':float(np.mean([r['auc'] for r in unseen])-np.mean(p0_vals)),
            'p0_all8_macro_auc':float(np.mean([p0_comparison[src]['p0_auc'] for src in SOURCES_GENIMAGE])),
            'delta_all8_macro_auc':float(np.mean([r['auc'] for r in per_source_rows])-np.mean([p0_comparison[src]['p0_auc'] for src in SOURCES_GENIMAGE])),
        }
    report={
        'purpose':'Existing GenImage Eval8000 benchmark; fixed Step2625 R-Debias; exploratory external check, not a new pristine held-out cohort',
        'dataset':'GenImage Eval8000','checkpoint':identity,
        'n_manifest_rows':8000,'n_unique_images':4500,
        'note':'Shared real images are reused across domains; do not treat 8000 rows as 8000 independent images; SD1.4 is training generator and excluded from unseen7.',
        'candidate':candidate,'unseen7_macro_auc':float(np.mean([r['auc'] for r in unseen])),
        'unseen7_worst_auc':float(worst['auc']), 'unseen7_worst_source':worst['source'],
        'unseen7_fake_top1_B1':float(np.mean([r['fake_top1_B1'] for r in unseen])),
        'unseen7_fake_top10_B1':float(np.mean([r['fake_top10_B1'] for r in unseen])),
        'macro_auc_all8':float(np.mean([r['auc'] for r in per_source_rows])),
        'difficult4_macro_auc':float(np.mean([r['auc'] for r in per_source_rows if r['source'] in HARD])),
        'per_source':{r['source']:r for r in per_source_rows},
        'p0_existing_per_source_comparison':p0_comparison,
        'p0_comparison_summary':p0_comparison_summary,
        'p0_same_image_comparison':p0_same_image,
    }
    save_report(out,report,df['global_index'].astype(str).tolist(),labels,z,scores,ids,per_source_rows,
                print_tag='EARLY9 R-DEBIAS GENIMAGE8000')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/EviRoute_edge'))
    p.add_argument('--dataset',choices=['pilot200','genimage8000'],required=True)
    p.add_argument('--checkpoint',type=Path,default=Path('distillation/p0_matched_pair_20k_v1/R_Debias/final_step.pt'))
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.batch_size<1 or args.workers<0 or not torch.cuda.is_available():
        p.error('batch-size positive, workers nonnegative, and CUDA required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=args.root.resolve()
    ckpt_path=absolute(root,args.checkpoint)
    out=absolute(root,args.out) if args.out else root/f'evaluation/early9_RDebias_{args.dataset}_v1'
    if out.exists():
        raise RuntimeError(f'Output directory exists; no overwrite allowed: {out}')
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    if ckpt_path.resolve() != (root/PAIR_DIR/'R_Debias'/'final_step.pt').resolve():
        raise RuntimeError('Only fixed R-Debias Step2625 final_step.pt may be evaluated')
    identity=check_checkpoint(ckpt,ckpt_path,root)
    print('R-DEBIAS MATCHED PAIR CHECKPOINT: PASS',flush=True)
    print('R-DEBIAS CHECKPOINT SHA256:',identity['checkpoint_sha256'],flush=True)
    source=root/'evaluation/eval_early9_p0.py'
    spec=importlib.util.spec_from_file_location('rdebias_original_evaluator',source)
    if spec is None or spec.loader is None:
        raise RuntimeError('Missing original Early9 PyTorch evaluator: '+str(source))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    module.P0_CKPT=ckpt_path  # In-memory override; original evaluator remains untouched
    model,cls,restored=module.load_model(torch.device('cuda:0'))
    if restored.get('paired_p0_20k',{}).get('arm') != ARM or restored['paired_p0_20k']['step'] != EXPECTED_STEP:
        raise RuntimeError('Model loader restored wrong arm or Step')
    val_cache=root/'distillation/depth_sweep_features/vitb_early9/val.pt'
    val=torch.load(val_cache,map_location='cpu',weights_only=False,mmap=True)
    val_paths=[str(p) for p in val['paths']]
    val_y=torch.as_tensor(val['labels']).cpu().numpy().astype(np.int64)
    val_z,_,_=check_tensors(*module.inference(model,cls,val_paths,torch.device('cuda:0'),args.batch_size,args.workers))
    val_auc=float(roc_auc_score(val_y,val_z))
    expected=identity['expected_val_auc']
    if abs(val_auc-expected)>1e-5:
        raise RuntimeError(f'Checkpoint reload validation AUC mismatch: actual={val_auc}, saved={expected}')
    print('FIXED STEP2625 VAL AUC RELOAD: PASS',f'{val_auc:.6f}',flush=True)
    if args.dataset=='pilot200':
        run_pilot(root,out,module,model,cls,torch.device('cuda:0'),args,identity)
    else:
        run_genimage(root,out,module,model,cls,torch.device('cuda:0'),args,identity)


if __name__=='__main__':
    main()
