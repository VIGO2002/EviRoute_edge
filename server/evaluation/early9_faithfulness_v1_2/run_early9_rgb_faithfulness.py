#!/usr/bin/env python3
"""Early-9 R-BCE vs R-Debias: within-model, ring-matched RGB faithfulness.

Fixed Step 2625. No training. Frozen original Top-10 masks remain canonical.
Strict logit audit with optional explicitly limited near-tie boundary swaps;
all permitted exceptions and sensitivity exclusions are recorded.

The perturbation copies exactly FGTS-main/make_causal_perturbations_clean_v2.py:
per selected 16x16 RGB patch, replace uint8 pixels with the mean of THAT PATCH.
Matching random patches are drawn from B1/B2-only/interior separately,
excluding all 10 selected patch indices. No reliance on cached feature tokens
for perturbed predictions: the original Early-9 inference code re-runs backbone.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

DOMAIN_ORDER=('ADM','BigGAN','Midjourney','VQDM','GLIDE','SD1.4','SD1.5','Wukong')
ARM_SPECS=(('R_BCE','R-BCE'),('R_Debias','R-Debias'))


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def zone(pid):
    pid=int(pid)
    if not 0<=pid<196:
        raise ValueError('Patch index out of bounds')
    r,c=divmod(pid,14)
    if r in (0,13) or c in (0,13): return 0
    if r in (1,12) or c in (1,12): return 1
    return 2


def matching_control(selected, rng):
    selected=[int(x) for x in selected]
    if len(selected)!=10 or len(set(selected))!=10:
        raise ValueError('Selected patches must be 10 distinct positions')
    s=set(selected)
    zones=Counter(zone(i) for i in selected)
    result=[]
    for j in range(3):
        pool=[i for i in range(196) if zone(i)==j and i not in s]
        if len(pool)<zones[j]:
            raise ValueError('Insufficient same-ring unselected control patches')
        result.extend(rng.sample(pool,zones[j]))
    if len(result)!=10 or s.intersection(result):
        raise AssertionError('Invalid random control')
    if Counter(zone(i) for i in result)!=zones:
        raise AssertionError('Failed exact ring match')
    return result


def local_mean_image(image224, patch_ids):
    """Exact FGTS uint8 replacement semantics; image224 already RGB 224x224."""
    if image224.mode!='RGB' or image224.size!=(224,224):
        raise ValueError('Expected RGB 224 x 224')
    a=np.array(image224).copy()
    for pid in patch_ids:
        r,c=divmod(int(pid),14)
        region=a[r*16:(r+1)*16,c*16:(c+1)*16]
        region[:]=region.reshape(-1,3).mean(axis=0)  # numpy uint8 truncation
    return Image.fromarray(a,'RGB')


def preview_selection(df, sample_n, seed):
    out=[]
    for domain in DOMAIN_ORDER:
        fake=df[(df.domain==domain)&(df.label==1)].sort_values('global_index')
        if len(fake)!=500:
            raise RuntimeError(f'Expected 500 fake images for {domain}')
        rr=np.random.default_rng(int(seed)+DOMAIN_ORDER.index(domain)*10007)
        pick=rr.choice(fake['global_index'].to_numpy(dtype=np.int64),size=sample_n,replace=False)
        out.extend(int(x) for x in sorted(pick))
    return sorted(out)


def source_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import '+str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def check_sources(root):
    path=root/'evaluation/eval_early9_p0.py'
    if not path.is_file():
        raise FileNotFoundError('Original Early-9 loading/inference code missing: '+str(path))
    for entry in ('evaluation/early9_pair_genimage_common_v1.py',
                  'evaluation/eval_early9_matched_pair_genimage.py'):
        if not (root/entry).is_file(): raise FileNotFoundError(str(root/entry))
    sys.path.insert(0,str(root/'evaluation'))
    import early9_pair_genimage_common_v1 as common
    import eval_early9_matched_pair_genimage as strictpair
    base=root/'distillation/p0_matched_pair_20k_v1'
    bpath=base/'R_BCE/final_step.pt';dpath=base/'R_Debias/final_step.pt'
    # Matching training lineage checks from the tested strict paired evaluator.
    dck=torch.load(dpath,map_location='cpu',weights_only=False)
    did=common.check_checkpoint(dck,dpath,root)
    bck=torch.load(bpath,map_location='cpu',weights_only=False)
    bid=strictpair.verify_bce_ckpt(bck,bpath,root,did)
    del dck,bck
    manifest=root/'evaluation/genimage_8domain_eval8000/manifest.csv'
    paired=root/'evaluation/early9_RBCE_vs_RDebias_genimage8000_v1'
    summary=json.loads((paired/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status')!='completed' or summary.get('n_manifest_rows')!=8000:
        raise RuntimeError('Strict paired summary status/count is invalid')
    if summary['eval_manifest_sha256']!=sha256(manifest):
        raise RuntimeError('Eval8000 manifest SHA differs from strict paired evaluation')
    for key,idobj in [('R_BCE_checkpoint',bid),('R_Debias_checkpoint',did)]:
        if summary[key]['checkpoint_sha256']!=idobj['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint mismatch: {key}')
    df,paths=common.read_genimage(root)
    with np.load(paired/'paired_outputs.npz',allow_pickle=False) as f:
        expect={'global_index','labels','R_BCE_logits','R_Debias_logits',
                'R_BCE_top10_indices','R_Debias_top10_indices'}
        if not expect.issubset(f.files):raise RuntimeError('Paired NPZ missing arrays')
        ix=f['global_index']; labels=f['labels']
        if not np.array_equal(ix,df.global_index.to_numpy(dtype=np.int64)) or not np.array_equal(labels,df.label.to_numpy(dtype=np.int64)):
            raise RuntimeError('Paired NPZ disagrees with Eval8000 manifest')
        cached={k:np.asarray(f[k]).copy() for k in expect}
    if set(df['domain'])!=set(DOMAIN_ORDER):raise RuntimeError('Domain set altered')
    return path,base,df,cached,{'R_BCE':bid,'R_Debias':did},sha256(paired/'paired_outputs.npz')


def audit_clean(outputs, targets_z, targets_idx, arm, gids,
                atol=0.002, max_near_ties=0, max_boundary_score_gap=0.0005):
    """Audit clean RGB inference against frozen strict-pair outputs.

    All logits retain the original strict tolerance. Top-10 rank permutations
    are recorded; a changed member normally aborts. Optional exceptions ONLY
    allow a single swap at ranks #10/#11 when the currently recomputed patch
    scores are nearly tied AND the top nine selected positions are unchanged.
    Always use the FROZEN top10 for the intervention (not a regenerated mask).
    """
    z, sc, idx = outputs
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    idx = np.asarray(idx, dtype=np.int64)
    targets_idx = np.asarray(targets_idx, dtype=np.int64)
    targets_z = np.asarray(targets_z, dtype=np.float64).reshape(-1)
    sc = np.asarray(sc, dtype=np.float64)
    gids = np.asarray(gids, dtype=np.int64).reshape(-1)
    n = len(gids)
    if max_near_ties < 0 or not np.isfinite(max_boundary_score_gap) or max_boundary_score_gap <= 0:
        raise ValueError('Invalid near-tie audit bounds')
    if (z.shape != (n,) or targets_z.shape != (n,) or
        idx.shape != (n, 10) or targets_idx.shape != (n, 10) or
        sc.shape != (n, 196)):
        raise RuntimeError(f'{arm}: original model inference output shapes are wrong')
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(sc)) or not np.all(np.isfinite(targets_z)):
        raise RuntimeError(f'{arm}: nonfinite clean scores')
    for name, arr in [('current', idx), ('frozen', targets_idx)]:
        if np.any(arr < 0) or np.any(arr >= 196) or np.any(np.diff(np.sort(arr, axis=1), axis=1) == 0):
            raise RuntimeError(f'{arm}: invalid/duplicated {name} Top-10 indices')
    # The unchanged historical logit tolerance: atol=0.002, rtol=1e-4.
    err = np.abs(z - targets_z)
    bad = np.flatnonzero(err > atol + 1e-4 * np.abs(targets_z))
    identical_order = np.all(idx == targets_idx, axis=1)
    identical_set = np.all(np.sort(idx, axis=1) == np.sort(targets_idx, axis=1), axis=1)
    order_changed = np.flatnonzero(~identical_order)
    membership_changed = np.flatnonzero(~identical_set)
    top1_changed = np.flatnonzero(idx[:, 0] != targets_idx[:, 0])

    exceptions = []
    failures = []
    for k in membership_changed:
        k=int(k)
        frozen, current = targets_idx[k], idx[k]
        old_only = sorted(int(i) for i in set(frozen) - set(current))
        new_only = sorted(int(i) for i in set(current) - set(frozen))
        gap = float(abs(sc[k, old_only[0]] - sc[k, new_only[0]])) if len(old_only)==len(new_only)==1 else None
        # The original Top-9 and new Top-9 must have identical selected sets;
        # the only accepted change is exactly the 10th slot.
        first_nine_same = bool(np.array_equal(np.sort(frozen[:9]), np.sort(current[:9])))
        old_rank = int(np.sum(sc[k] > sc[k, old_only[0]]) + 1) if len(old_only)==1 else None
        new_rank = int(np.sum(sc[k] > sc[k, new_only[0]]) + 1) if len(new_only)==1 else None
        record = {'global_index':int(gids[k]),'logit_abs_error':float(err[k]),
                  'old_only':old_only,'new_only':new_only,
                  'old_rank_by_current_scores':old_rank,'new_rank_by_current_scores':new_rank,
                  'current_score_gap':gap,'first_nine_set_same':first_nine_same,
                  'frozen_top10':frozen.tolist(),'new_top10':current.tolist()}
        near = (len(old_only)==len(new_only)==1 and first_nine_same and
                idx[k,0]==targets_idx[k,0] and gap is not None and
                old_rank is not None and old_rank <= 12 and
                gap <= max_boundary_score_gap)
        if near:
            exceptions.append(record)
        else:
            failures.append(record)

    if exceptions:
        print(f'{arm}: near-tie boundary-swap candidates (FROZEN masks preserved): '
              f'{json.dumps(exceptions, ensure_ascii=False)}',flush=True)
    if bad.size or failures or len(exceptions)>max_near_ties:
        bad_union=sorted(set(bad.tolist()) | set(top1_changed.tolist()))
        details=[{'global_index':int(gids[k]), 'logit_abs_error':float(err[k]),
                  'frozen_top10':targets_idx[k].tolist(), 'new_top10':idx[k].tolist()}
                 for k in bad_union[:8]]
        raise RuntimeError(f'{arm} ORIGINAL NUMERICAL AUDIT FAIL: max_abs_logit_error={float(np.max(err)):.8g}, '
            f'bad_logits={len(bad)}, changed_top10_order={len(order_changed)}, '
            f'changed_top10_membership={len(membership_changed)}, changed_top1={len(top1_changed)}, '
            f'near_ties_eligible={len(exceptions)}, explicitly_allowed={max_near_ties}, '
            f'gap_tolerance={max_boundary_score_gap}; first_mismatches={failures[:8] or details}. '
            'STOP: do not interpret perturbation results.')

    report = {'n':n, 'max_abs_logit_error':float(np.max(err)),
              'mean_abs_logit_error':float(np.mean(err)),
              'top10_set_exact_match':len(membership_changed)==0,
              'top10_set_match_count':int(n-len(membership_changed)),
              'top10_rank_order_changed_count':int(len(order_changed)),
              'top1_index_changed_count':int(len(top1_changed)),
              'top10_rank_order_changed_global_indices':gids[order_changed].astype(int).tolist(),
              'top10_membership_changed_global_indices':gids[membership_changed].astype(int).tolist(),
              'allowed_near_tie_count':len(exceptions),
              'allowed_near_tie_details':exceptions,
              'max_near_ties_argument':int(max_near_ties),
              'max_boundary_score_gap_argument':float(max_boundary_score_gap),
              'selection_used_for_perturbation':'frozen strict-pair Top10 indices'}
    if len(order_changed):
        print(f'{arm}: NOTE {len(order_changed)}/{n} Top-10 ordered-index changes; '
              f'{len(membership_changed)} permitted near-tie set changes; '
              f'global_index={report["top10_rank_order_changed_global_indices"][:30]}', flush=True)
    return report


def build_perturb_paths(rows, arm, cache, num_random, seed, temp_dir):
    """Returns paths + per-image metadata; clean original handled separately."""
    paths=[]; plan=[]
    for row in rows:
        gid=int(row['global_index'])
        chosen=[int(v) for v in cache[f'{arm}_top10_indices'][gid]]
        # This is a checked protocol equivalence to original old perturb_image_224.
        with Image.open(row['path']) as im:
            original=im.convert('RGB').resize((224,224),resample=Image.Resampling.BICUBIC)
        block_seed=int.from_bytes(hashlib.sha256(f'{seed}|{arm}|{gid}'.encode()).digest()[:8],'big')
        rng=random.Random(block_seed)
        variants=[('selected',0,chosen)]
        variants.extend(('control',j,matching_control(chosen,rng)) for j in range(num_random))
        for tag,j,pids in variants:
            p=Path(temp_dir)/f'{arm}_{gid:05d}_{tag}_{j:02d}.png'
            local_mean_image(original,pids).save(p,format='PNG')
            paths.append(str(p))
            plan.append((gid,tag,j,pids))
    return paths,plan


def evaluate_arm(root, model_source, ckpt_path, arm, rows, cache,
                 device, batch_size, workers, chunk_size, num_random, seed,
                 max_near_ties=0, max_boundary_score_gap=0.0005):
    mod=source_module(model_source,'early9_faithfulness_orig_'+arm)
    mod.P0_CKPT=ckpt_path
    model,cls,loaded=mod.load_model(device)
    try:
        if loaded.get('paired_p0_20k',{}).get('arm')!=arm.replace('_','-') or loaded['paired_p0_20k'].get('step')!=2625:
            raise RuntimeError('Loaded model identity/step mismatch for '+arm)
        model.eval()
        # First perform FULL clean audit for exactly the chosen original image paths.
        clean_paths=[str(r['path']) for r in rows]
        gids=np.array([int(r['global_index']) for r in rows],dtype=np.int64)
        clean=mod.inference(model,cls,clean_paths,device,batch_size,workers)
        audit=audit_clean(clean,cache[f'{arm}_logits'][gids],cache[f'{arm}_top10_indices'][gids],arm,gids,
                          max_near_ties=max_near_ties, max_boundary_score_gap=max_boundary_score_gap)
        print(arm,'ORIGINAL RGB / FROZEN PAIRED LOGITS + TOP10: PASS',audit,flush=True)
        clean_map={int(g):float(z) for g,z in zip(gids,np.asarray(clean[0]).reshape(-1))}
        results=[]
        for start in range(0,len(rows),chunk_size):
            sub=rows[start:start+chunk_size]
            with tempfile.TemporaryDirectory(prefix='early9_faith_',dir=None) as td:
                paths,plan=build_perturb_paths(sub,arm,cache,num_random,seed,td)
                logits,scores,top10=mod.inference(model,cls,paths,device,batch_size,workers)
                logits=np.asarray(logits,dtype=np.float64).reshape(-1)
                if len(logits)!=len(plan) or not np.isfinite(logits).all():
                    raise RuntimeError('Perturbed inference produced wrong/nonfinite outputs')
                grouped={int(r['global_index']):{'selected':None,'controls':[]} for r in sub}
                for (gid,kind,idx,ids),v in zip(plan,logits):
                    if kind=='selected':
                        if grouped[gid]['selected'] is not None:raise RuntimeError('Duplicate selected perturbation')
                        grouped[gid]['selected']=float(v)
                    else:grouped[gid]['controls'].append(float(v))
                for r in sub:
                    gid=int(r['global_index']); d=grouped[gid]
                    if d['selected'] is None or len(d['controls'])!=num_random:
                        raise RuntimeError('Missing inference variant')
                    orig=clean_map[gid];dsel=orig-d['selected']
                    controls=[orig-v for v in d['controls']]
                    adv=float(dsel-np.mean(controls))
                    results.append({'global_index':gid,'domain':r['domain'],'label':1,
                        'path':r['path'],'arm':arm,'clean_logit':orig,
                        'selected_logit':d['selected'],'d_sel':float(dsel),
                        'mean_d_ctrl':float(np.mean(controls)),
                        'advantage':adv,'win':int(adv>0),
                        'fake_correct':int(orig>=0),
                        'top10_indices':json.dumps([int(x) for x in cache[f'{arm}_top10_indices'][gid]]),
                        'control_drops':json.dumps([float(x) for x in controls])})
            print(f'{arm} perturbation progress: {min(start+chunk_size,len(rows))}/{len(rows)}',flush=True)
        return audit,results
    finally:
        del model,cls,mod
        if torch.cuda.is_available():torch.cuda.empty_cache()


def mean_stat(rows):
    n=len(rows)
    if n==0:return {'n':0}
    return {'n':n,'advantage':float(np.mean([r['advantage'] for r in rows])),
        'win_rate':float(np.mean([r['win'] for r in rows])),
        'mean_d_sel':float(np.mean([r['d_sel'] for r in rows])),
        'mean_d_ctrl':float(np.mean([r['mean_d_ctrl'] for r in rows])),
        'fake_correct_rate':float(np.mean([r['fake_correct'] for r in rows]))}


def summarize(rows):
    out={}
    for arm,_ in ARM_SPECS:
        part=[r for r in rows if r['arm']==arm]
        out[arm]={'all8':mean_stat(part),
                  'unseen7':mean_stat([r for r in part if r['domain']!='SD1.4']),
                  'unseen7_correct_fake':mean_stat([r for r in part if r['domain']!='SD1.4' and r['fake_correct']]),
                  'per_source':{d:mean_stat([r for r in part if r['domain']==d]) for d in DOMAIN_ORDER}}
    return out


def bootstrap_macro(rows,replicates,seed):
    """Paired same-image, within-source resampling of two arms; source-fixed CIs."""
    rng=np.random.default_rng(seed)
    lookup={(arm,d):{r['global_index']:r for r in rows if r['arm']==arm and r['domain']==d}
            for arm,_ in ARM_SPECS for d in DOMAIN_ORDER}
    aligned={}
    for d in DOMAIN_ORDER:
        gids=sorted(lookup[('R_BCE',d)])
        if not gids or gids!=sorted(lookup[('R_Debias',d)]):
            raise RuntimeError('R-BCE / R-Debias sample matching failed for '+d)
        aligned[d]={a:{key:np.array([lookup[(a,d)][gid][key] for gid in gids],dtype=np.float64)
                       for key in ('advantage','win')}
                    for a,_ in ARM_SPECS}
    values={}
    for metric in ('advantage','win'):
        for setname,ds in [('all8',DOMAIN_ORDER),
                           ('unseen7',tuple(d for d in DOMAIN_ORDER if d!='SD1.4'))]:
            results={k:np.empty(replicates,dtype=np.float64)
                     for k in ('R_BCE','R_Debias','R_Debias_minus_R_BCE')}
            for rep in range(replicates):
                each={'R_BCE':[],'R_Debias':[]}
                for d in ds:
                    n=len(aligned[d]['R_BCE'][metric])
                    pick=rng.integers(0,n,n)
                    for arm,_ in ARM_SPECS:
                        each[arm].append(float(np.mean(aligned[d][arm][metric][pick])))
                for arm,_ in ARM_SPECS:
                    results[arm][rep]=float(np.mean(each[arm]))
                results['R_Debias_minus_R_BCE'][rep]=results['R_Debias'][rep]-results['R_BCE'][rep]
            points={arm:float(np.mean([np.mean(aligned[d][arm][metric]) for d in ds]))
                    for arm,_ in ARM_SPECS}
            points['R_Debias_minus_R_BCE']=points['R_Debias']-points['R_BCE']
            for key,x in results.items():
                values[f'{setname}_{key}_{metric}']={'point':points[key],
                    'ci95_low':float(np.percentile(x,2.5)),
                    'ci95_high':float(np.percentile(x,97.5))}
    return {'method':'paired fake-image within-source bootstrap; same selected image draws for both models; generator sources fixed',
        'replicates':replicates,'seed':seed,'estimates':values,
        'caution':'Exploratory evaluation conditional on previously inspected GenImage Eval8000; CIs do not cover model or generator uncertainty. Comparing raw logit advantages across models does not normalize logit scale.'}


def clean_preflight(model_source, ckpt_root, rows, cache, device, batch_size, workers,
                    max_near_ties=0, max_boundary_score_gap=0.0005):
    """Run both model clean-only audits BEFORE costly RGB perturbations."""
    paths = [str(r['path']) for r in rows]
    gids = np.asarray([int(r['global_index']) for r in rows], dtype=np.int64)
    for arm, _ in ARM_SPECS:
        mod = source_module(model_source, 'early9_faith_preflight_' + arm)
        mod.P0_CKPT = ckpt_root / arm / 'final_step.pt'
        model, cls, loaded = mod.load_model(device)
        try:
            if (loaded.get('paired_p0_20k', {}).get('arm') != arm.replace('_', '-') or
                loaded['paired_p0_20k'].get('step') != 2625):
                raise RuntimeError('Loaded model identity/step mismatch in preflight: '+arm)
            model.eval()
            outputs = mod.inference(model, cls, paths, device, batch_size, workers)
            checked = audit_clean(outputs, cache[f'{arm}_logits'][gids],
                                  cache[f'{arm}_top10_indices'][gids], arm, gids,
                                  max_near_ties=max_near_ties, max_boundary_score_gap=max_boundary_score_gap)
            print(arm, 'CLEAN PREFLIGHT: PASS', checked, flush=True)
        finally:
            del model, cls, mod
            if torch.cuda.is_available():torch.cuda.empty_cache()
    print('BOTH ARMS CLEAN PREFLIGHT: PASS', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/EviRoute_edge'))
    p.add_argument('--out',type=Path,default=Path('evaluation/early9_rgb_faithfulness_pilot_v1'))
    p.add_argument('--samples-per-source',type=int,default=2)
    p.add_argument('--num-random',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261009)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--chunk-size',type=int,default=8)
    p.add_argument('--preflight-only', action='store_true', help='Both-arm clean numerical + boundary-swap audit, no perturbations')
    p.add_argument('--max-near-ties',type=int,default=0,help='Explicitly permit at most this many VERIFIED rank10 boundary swaps per arm; 0 = strict')
    p.add_argument('--max-boundary-score-gap',type=float,default=0.0005,help='Maximum absolute scorer-unit difference for a permissible #10/#11 boundary swap')
    args=p.parse_args()
    if not (1<=args.samples_per_source<=500 and 1<=args.num_random<=100 and args.bootstrap>=100 and
            args.batch_size>=1 and args.workers>=0 and args.chunk_size>=1 and
            args.max_near_ties>=0 and np.isfinite(args.max_boundary_score_gap) and args.max_boundary_score_gap>0):
        p.error('Invalid experimental arguments')
    if not torch.cuda.is_available():p.error('CUDA required: original model + full RGB re-inference')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    root=args.root.resolve();out=args.out if args.out.is_absolute() else root/args.out
    if out.exists() and not args.preflight_only:raise RuntimeError('Output exists; refusing overwrite: '+str(out))
    model_source,ckpt_root,df,cache,identities,paired_npz_sha=check_sources(root)
    gids=preview_selection(df,args.samples_per_source,args.seed)
    rows=[df.iloc[gid].to_dict() for gid in gids]
    if not all(r['label']==1 for r in rows):raise RuntimeError('Selection included a real image')
    print('SAMPLE SELECTION:',[(d,sum(x['domain']==d for x in rows)) for d in DOMAIN_ORDER],flush=True)
    print('CHECKPOINT LINEAGE + MANIFEST: PASS',flush=True)
    print('PAIRED OUTPUTS SHA256:',paired_npz_sha,flush=True)
    print('ORIGINAL LOADER SHA256:',sha256(model_source),flush=True)
    records=[]; clean_checks={}
    device=torch.device('cuda:0')
    if args.preflight_only:
        clean_preflight(model_source,ckpt_root,rows,cache,device,args.batch_size,args.workers,
                        args.max_near_ties,args.max_boundary_score_gap)
        return
    for arm,_ in ARM_SPECS:
        ckpt=ckpt_root/arm/'final_step.pt'
        clean_checks[arm],result=evaluate_arm(root,model_source,ckpt,arm,rows,cache,
            device,args.batch_size,args.workers,args.chunk_size,args.num_random,args.seed,
            args.max_near_ties,args.max_boundary_score_gap)
        records.extend(result)
    ambiguous=sorted(set().union(*(set(clean_checks[a]['top10_membership_changed_global_indices']) for a,_ in ARM_SPECS)))
    sensitivity_records=[r for r in records if r['global_index'] not in set(ambiguous)]
    report={'status':'completed','purpose':'Early-9 original-input-space within-model Top10 perturbation vs exact ring matched random control',
        'mode':'exploratory; not a new held-out evaluation; no training / model selection',
        'root':str(root),'parameters':vars(args)|{'root':str(root),'out':str(out)},
        'n_unique_fake_images':len(rows),'selected_global_indices':gids,
        'provenance':{'paired_npz_sha256':paired_npz_sha,'original_model_loader_sha256':sha256(model_source),
                      'checkpoints':{a:identities[a]['checkpoint_sha256'] for a,_ in ARM_SPECS}},
        'clean_output_checks':clean_checks,
        'boundary_tie_sensitivity':{
            'excluded_global_indices':ambiguous,
            'n_unique_excluded':len(ambiguous),
            'results':summarize(sensitivity_records),
            'bootstrap_ci95':bootstrap_macro(sensitivity_records,args.bootstrap,args.seed+11)},
        'method':{'perturbation':'uint8 per-patch 16x16 RGB own local mean, before standard original inference preprocessing',
            'controls':'each control excludes selected Top10, exact match B1/B2-only/interior counts',
            'topk':'within-model, ALWAYS original frozen strict-pair Top10 even if a verified near-tie swaps at rank10',
            'advantage':'(clean - selected logit) - mean_over_controls(clean - random-matched logit)',
            'win':'advantage > 0; one value per image','clean_tolerance':'atol=0.002 + rtol=0.0001; explicit limited near-tie swaps only when top9 unchanged and current-score gap below max_boundary_score_gap'},
        'results':summarize(records),'bootstrap_ci95':bootstrap_macro(records,args.bootstrap,args.seed+1)}
    out.mkdir(parents=True,exist_ok=False)
    with (out/'per_image_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    with (out/'selection_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['global_index','domain','path','label']);w.writeheader()
        w.writerows([{k:r[k] for k in ('global_index','domain','path','label')} for r in rows])
    for arm,_ in ARM_SPECS:
        for domain in ('unseen7','all8'):
            info=report['results'][arm][domain]
            print(f'{arm} {domain}: n={info["n"]}, advantage={info["advantage"]:.5f}, win={info["win_rate"]:.1%}',flush=True)
    print('RGB FAITHFULNESS AUDIT: PASS',flush=True)
    print('SAVED:',out,flush=True)

if __name__=='__main__':main()
