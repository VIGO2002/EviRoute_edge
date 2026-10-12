#!/usr/bin/env python3
"""Paired class-aligned low-level characterization: Early9 R-BCE vs R-Debias.

Reuses exact original v2 RGB metric computation and percentile ranks (five metrics).
Reports the user-requested four-metric Detail4 separately from legacy five-metric
composite. Fake selects frozen strict-pair Top10; Real selects -fake-score Top10.
Ring-matched blue patches are lowest class-aligned support in each exact ring.
No model training. Explored GenImage Eval8000, NOT pristine final held-out.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import legacy_analyze_lowlevel_red_blue_v2 as legacy

ARMS=('R_BCE','R_Debias')
DOMAINS=('ADM','BigGAN','Midjourney','VQDM','GLIDE','SD1.4','SD1.5','Wukong')
METRICS4=('lap_var','grad_energy','residual_abs','color_var')
METRICS5=('lap_var','grad_energy','residual_abs','fft_hf_ratio','color_var')
CACHE_NAME='patch_scores.npz'
META_NAME='patch_scores_meta.json'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()


def zone(i):
    i=int(i)
    if not 0<=i<196: raise ValueError('Bad patch id')
    r,c=divmod(i,14)
    if r in (0,13) or c in (0,13): return 0
    if r in (1,12) or c in (1,12): return 1
    return 2


def red_blue(scores, label, fake_frozen=None):
    """Return high class-aligned support and exact-ring low support control."""
    scores=np.asarray(scores,dtype=np.float64)
    if scores.shape!=(196,) or not np.isfinite(scores).all() or label not in (0,1):
        raise ValueError('Bad score/label')
    class_scores=scores if label==1 else -scores
    if label==1:
        if fake_frozen is None: raise ValueError('Fake requires frozen Top10')
        red=np.asarray(fake_frozen,dtype=np.int64).tolist()
    else:
        red=np.argsort(-class_scores,kind='stable')[:10].astype(int).tolist()
    if len(red)!=10 or len(set(red))!=10 or min(red)<0 or max(red)>=196:
        raise ValueError('Bad evidence indices')
    redset=set(red);counts=Counter(zone(i) for i in red)
    blue=[]
    for ring in range(3):
        pool=[i for i in range(196) if zone(i)==ring and i not in redset]
        ranked=sorted(pool,key=lambda i:(float(class_scores[i]),i))
        if len(ranked)<counts[ring]: raise ValueError('Insufficient same-ring controls')
        blue.extend(ranked[:counts[ring]])
    if len(blue)!=10 or set(red)&set(blue): raise AssertionError('Bad blue set')
    if Counter(zone(i) for i in red)!=Counter(zone(i) for i in blue):
        raise AssertionError('Inexact ring match')
    return red,blue


def choose_rows(df, faith_summary, fake_count, real_count, seed):
    """Nested deterministic subset of frozen 240 faithfulness fake IDs + unique real IDs."""
    if faith_summary.get('status')!='completed' or faith_summary.get('n_unique_fake_images')!=240:
        raise RuntimeError('Faithfulness 240 source incomplete')
    fconf=faith_summary.get('parameters',{})
    if int(fconf.get('samples_per_source',-1))!=30 or int(fconf.get('seed',-1))!=20261009:
        raise RuntimeError('Faithfulness frozen selection provenance unexpected')
    frozen=[int(x) for x in faith_summary['selected_global_indices']]
    if len(frozen)!=240 or len(set(frozen))!=240: raise RuntimeError('Duplicate faithfulness ID')
    fake=[]
    for d in DOMAINS:
        group=[i for i in frozen if str(df.iloc[i]['domain'])==d and int(df.iloc[i]['label'])==1]
        if len(group)!=30: raise RuntimeError(f'Faithfulness cohort not 30 fake for {d}')
        fake.extend(sorted(group)[:fake_count])
    # All eight domains share the same 500 real paths. Use ONE unique cohort only.
    pool=df[(df.domain=='ADM')&(df.label==0)].sort_values('global_index')
    if len(pool)!=500 or pool.path.nunique()!=500: raise RuntimeError('Shared real 500 mismatch')
    rng=np.random.default_rng(seed+7701)
    real=sorted(int(i) for i in rng.choice(pool.global_index.to_numpy(dtype=np.int64),size=real_count,replace=False))
    select=fake+real
    out=df.iloc[select].copy().reset_index(drop=True)
    out['cohort']=['fake']*len(fake)+['real_shared']*len(real)
    if out.path.nunique()!=len(out): raise RuntimeError('Repeated physical file in Detail4 selection')
    return out


def load_project(root):
    sys.path.insert(0,str(root/'evaluation'))
    sys.path.insert(0,str(root/'evaluation/early9_faithfulness_v1_2'))
    import run_early9_rgb_faithfulness as fw
    model_source,ckpt_root,df,frozen,identities,paired_hash=fw.check_sources(root)
    faith_path=root/'evaluation/early9_rgb_faithfulness_240_v1_2/summary.json'
    faith=json.loads(faith_path.read_text(encoding='utf-8'))
    if faith.get('provenance',{}).get('paired_npz_sha256')!=paired_hash:
        raise RuntimeError('Faithfulness cohort source differs from frozen strict-pair')
    for arm in ARMS:
        if faith['provenance']['checkpoints'][arm]!=identities[arm]['checkpoint_sha256']:
            raise RuntimeError('Faithfulness checkpoint lineage differs')
    return fw,model_source,ckpt_root,df,frozen,identities,paired_hash,faith,sha256(faith_path)


def create_cohort_manifest(out, rows):
    with (out/'selection_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['row_index','global_index','domain','cohort','label','path'])
        w.writeheader()
        for ix,row in rows.iterrows():
            w.writerow({'row_index':ix,'global_index':int(row.global_index),'domain':row.domain,
                        'cohort':row.cohort,'label':int(row.label),'path':row.path})


def do_extract(args,root,out):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA device required for full patch-score extraction')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    fw,source,ckroot,df,frozen,ids,pairsha,faith,faithsha=load_project(root)
    rows=choose_rows(df,faith,args.fake_per_source,args.real_count,args.seed)
    if out.exists():raise RuntimeError(f'Refusing overwrite: {out}')
    print('COHORT:',Counter((r.cohort,r.domain if r.label else 'REAL_SHARED') for r in rows.itertuples()),flush=True)
    gids=rows.global_index.to_numpy(dtype=np.int64)
    labels=rows.label.to_numpy(dtype=np.int64)
    paths=rows.path.astype(str).tolist()
    arrays={'global_index':gids,'label':labels}
    audits={}
    device=torch.device('cuda:0')
    for arm in ARMS:
        model_mod=fw.source_module(source,'detail4_early9_'+arm)
        model_mod.P0_CKPT=ckroot/arm/'final_step.pt'
        model,cls,loaded=model_mod.load_model(device)
        try:
            if loaded.get('paired_p0_20k',{}).get('arm')!=arm.replace('_','-') or loaded['paired_p0_20k'].get('step')!=2625:
                raise RuntimeError('Wrong Step2625 model loaded: '+arm)
            model.eval()
            with torch.inference_mode():
                z,scores,top10=model_mod.inference(model,cls,paths,device,args.batch_size,args.workers)
            z=np.asarray(z,dtype=np.float32).reshape(-1)
            scores=np.asarray(scores,dtype=np.float32)
            top10=np.asarray(top10,dtype=np.int64)
            if z.shape!=(len(rows),) or scores.shape!=(len(rows),196) or top10.shape!=(len(rows),10) or not np.isfinite(scores).all():
                raise RuntimeError('Wrong model output shape or NaNs')
            if not np.allclose(z,np.take_along_axis(scores,top10,axis=1).mean(axis=1),atol=0.002,rtol=1e-4):
                raise RuntimeError('Current image logit != own Top10 scores')
            refz=frozen[arm+'_logits'][gids]
            refix=frozen[arm+'_top10_indices'][gids]
            fake=labels==1
            fa=fw.audit_clean((z[fake],scores[fake],top10[fake]),refz[fake],refix[fake],arm,gids[fake],
                              max_near_ties=args.max_near_ties,max_boundary_score_gap=args.max_boundary_score_gap)
            # Real class support is -score; we still audit original fake-logit numerics.
            re=~fake
            err=np.abs(z[re].astype(np.float64)-refz[re].astype(np.float64))
            bad=err>(0.002+1e-4*np.abs(refz[re]))
            if bad.any():
                where=int(np.flatnonzero(bad)[0]); raise RuntimeError(f'{arm} REAL logit audit failed at {int(gids[re][where])}')
            real_top1_match=int(np.sum(top10[re,0]==refix[re,0]))
            real_set_match=int(np.sum(np.all(np.sort(top10[re],axis=1)==np.sort(refix[re],axis=1),axis=1)))
            audits[arm]={'fake':fa,'real':{'n':int(re.sum()),'max_abs_logit_error':float(err.max()),
                      'top1_match_count':real_top1_match,'fake_score_top10_set_match_count':real_set_match}}
            arrays[arm+'_logits']=z
            arrays[arm+'_scores']=scores
            arrays[arm+'_current_fake_top10']=top10.astype(np.int16)
            print(f'{arm} ORIGINAL AUDIT: PASS; fake {fa["top10_set_match_count"]}/{fa["n"]} set matches; '
                  f'real {real_set_match}/{int(re.sum())} set matches; max real logit error={float(err.max()):.8f}',flush=True)
        finally:
            del model,cls,model_mod
            torch.cuda.empty_cache()
    out.mkdir(parents=True,exist_ok=False)
    create_cohort_manifest(out,rows)
    np.savez_compressed(out/CACHE_NAME,**arrays)
    meta={'status':'extracted','experiment':'Early9 paired class-aligned low-level characterization',
          'caution':'EXPLORATORY GenImage Eval8000, not pristine held-out',
          'parameters':{'fake_per_source':args.fake_per_source,'real_count':args.real_count,'seed':args.seed,
                        'batch_size':args.batch_size,'workers':args.workers},
          'provenance':{'paired_npz_sha256':pairsha,'faith_summary_sha256':faithsha,
                        'legacy_metric_source_sha256':sha256(Path(__file__).with_name('legacy_analyze_lowlevel_red_blue_v2.py')),
                        'loader_sha256':sha256(source),
                        'checkpoints':{a:ids[a]['checkpoint_sha256'] for a in ARMS},
                        'manifest_sha256':sha256(root/'evaluation/genimage_8domain_eval8000/manifest.csv')},
          'audits':audits, 'n_fake':int((labels==1).sum()),'n_real_unique':int((labels==0).sum())}
    (out/META_NAME).write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PATCH SCORES EXTRACTION: PASS',flush=True)
    print('SAVED CACHE:',out/CACHE_NAME,flush=True)


def contrast_record(gid,domain,label,cohort,path,arm,scores,red,blue,metric_vals,logit):
    record={'global_index':int(gid),'domain':str(domain) if label==1 else 'REAL_SHARED',
            'label':int(label),'cohort':cohort,'path':str(path),'arm':arm,'clean_logit':float(logit),
            'top10_indices':'|'.join(map(str,red)),'blue_indices':'|'.join(map(str,blue)),
            'mean_class_score_gap':float(np.mean((scores if label else -scores)[red])-np.mean((scores if label else -scores)[blue]))}
    pct_deltas={}
    for metric in METRICS5:
        x=metric_vals[metric]
        percentile=legacy.percentile_ranks(x)
        a=float(np.mean(percentile[red]));b=float(np.mean(percentile[blue]))
        record[metric+'_percentile_delta']=a-b
        record[metric+'_raw_delta']=float(np.mean(x[red])-np.mean(x[blue]))
        pct_deltas[metric]=a-b
    record['detail4_percentile_delta']=float(np.mean([pct_deltas[m] for m in METRICS4]))
    record['legacy5_percentile_delta']=float(np.mean([pct_deltas[m] for m in METRICS5]))
    return record


def stats(vals):
    vals=np.asarray(vals,dtype=np.float64)
    if not len(vals):return {'n':0}
    return {'n':int(len(vals)),'mean':float(vals.mean()),'median':float(np.median(vals)),
            'negative_rate':float(np.mean(vals<0)), 'positive_rate':float(np.mean(vals>0))}


def get_group(records,label,domain=None):
    return {arm:{r['global_index']:r for r in records if r['arm']==arm and r['label']==label and (domain is None or r['domain']==domain)} for arm in ARMS}


def macro_bootstrap(group,domains,keys,reps,seed):
    rng=np.random.default_rng(seed)
    arr={}
    for d in domains:
        pair=get_group(group,1,d) if d!='REAL_SHARED' else get_group(group,0)
        gids=sorted(pair['R_BCE'])
        if not gids or gids!=sorted(pair['R_Debias']):
            raise RuntimeError('Detail4 paired image mismatch in domain '+d)
        arr[d]={a:{key:np.array([pair[a][gid][key] for gid in gids],dtype=np.float64) for key in keys} for a in ARMS}
    result={}
    for key in keys:
        draws={a:np.zeros(reps,dtype=np.float64) for a in (*ARMS,'delta_Debias_minus_BCE')}
        values={a:float(np.mean([np.mean(arr[d][a][key]) for d in domains])) for a in ARMS}
        values['delta_Debias_minus_BCE']=values['R_Debias']-values['R_BCE']
        for j in range(reps):
            ds={a:[] for a in ARMS}
            for d in domains:
                n=len(arr[d]['R_BCE'][key]); pick=rng.integers(0,n,size=n)
                for a in ARMS:ds[a].append(float(np.mean(arr[d][a][key][pick])))
            for a in ARMS:draws[a][j]=float(np.mean(ds[a]))
            draws['delta_Debias_minus_BCE'][j]=draws['R_Debias'][j]-draws['R_BCE'][j]
        result[key]={a:{'point':values[a], 'ci95_low':float(np.percentile(draws[a],2.5)),
                        'ci95_high':float(np.percentile(draws[a],97.5))}
                     for a in draws}
    return result


def write_rows(path,rows):
    if not rows:raise ValueError('Empty output rows')
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def do_analyze(args,root,out):
    if not (out/META_NAME).is_file() or not (out/CACHE_NAME).is_file():
        raise FileNotFoundError('Extract GPU scores first; missing cache or metadata')
    if (out/'summary.json').exists():raise RuntimeError('Summary exists; refusing overwrite')
    meta=json.loads((out/META_NAME).read_text())
    if meta.get('status')!='extracted':raise RuntimeError('Invalid cache status')
    fw,source,ckroot,df,frozen,ids,pairsha,faith,faithsha=load_project(root)
    proof=meta['provenance']
    if (proof['paired_npz_sha256']!=pairsha or proof['faith_summary_sha256']!=faithsha or
        proof['loader_sha256']!=sha256(source) or
        proof['legacy_metric_source_sha256']!=sha256(Path(__file__).with_name('legacy_analyze_lowlevel_red_blue_v2.py')) or
        proof['manifest_sha256']!=sha256(root/'evaluation/genimage_8domain_eval8000/manifest.csv') or
        any(proof['checkpoints'][a]!=ids[a]['checkpoint_sha256'] for a in ARMS)):
        raise RuntimeError('Cached score lineage no longer matches sources')
    with np.load(out/CACHE_NAME,allow_pickle=False) as z:
        ix=z['global_index'].astype(np.int64);label=z['label'].astype(np.int64)
        rows=df.iloc[ix].copy().reset_index(drop=True)
        if len(set(ix.tolist()))!=len(ix) or not np.array_equal(rows.global_index.to_numpy(dtype=np.int64),ix) or not np.array_equal(rows.label.to_numpy(dtype=np.int64),label):
            raise RuntimeError('Score cache not aligned to manifest')
        rcache={a:{'scores':z[a+'_scores'].copy(),'logits':z[a+'_logits'].copy()} for a in ARMS}
    frozen_fake={a:frozen[a+'_top10_indices'] for a in ARMS}
    records=[]
    for i,row in rows.iterrows():
        with Image.open(str(row.path)) as img:
            rgb=img.convert('RGB')
            computed=legacy.compute_patch_metrics(rgb)
        for arm in ARMS:
            scores=rcache[arm]['scores'][i]
            sel=frozen_fake[arm][int(row.global_index)] if int(row.label)==1 else None
            red,blue=red_blue(scores,int(row.label),sel)
            r=contrast_record(int(row.global_index),row.domain,int(row.label),
                               'fake' if int(row.label) else 'real_shared',row.path,arm,scores,red,blue,
                               computed,rcache[arm]['logits'][i])
            records.append(r)
        if (i+1)%50==0 or i+1==len(rows):
            print(f'Detail4 RGB metrics: {i+1}/{len(rows)} unique images',flush=True)
    write_rows(out/'per_image_detail4.csv',records)
    keys=['detail4_percentile_delta','legacy5_percentile_delta']+[m+'_percentile_delta' for m in METRICS5]
    main_groups={'fake_all8':list(DOMAINS),'fake_unseen7':[d for d in DOMAINS if d!='SD1.4'],
                 'real_unique':['REAL_SHARED']}
    report={'status':'completed','purpose':'Class-aligned high support vs exact-ring lowest support low-level characterization',
      'mode':'EXPLORATORY Eval8000 (same repeatedly inspected test images); NOT pixel-level artifact localization accuracy',
      'n_unique_images':len(rows),'n_fake':int((label==1).sum()),'n_real_unique':int((label==0).sum()),
      'definition':{'detail4':list(METRICS4),'legacy5':list(METRICS5),
         'original_source':'legacy_analyze_lowlevel_red_blue_v2.py',
         'percentile':'exact original within-image np.argsort rank / 195; each metric computed from RGB resized 224x224',
         'red_fake':'frozen strict-pair own Top10 highest fake-support patch scores',
         'red_real':'10 lowest fake-support patch scores, i.e. highest -fake-score (real-aligned) patches',
         'blue':'lowest class-aligned score patches disjoint from red with identical B1/B2-only/interior counts',
         'contrast':'mean percentile(rank(high/red))-mean percentile(rank(low/blue)); negative=red smoother/lower-detail',
         'caveat':'Original v2 script computes 5-metric composite including FFT; Detail4 is an explicitly separate 4-metric projection'},
      'provenance':meta['provenance'],'extraction_audits':meta['audits'],
      'bootstrap':{'replicates':args.bootstrap,'seed':args.seed+901,'method':'same-image paired within-source; fixed generator sources; real images sampled uniquely once'},
      'groups':{}}
    source_rows=[]
    for group,ds in main_groups.items():
        selected=[r for r in records if (r['label']==0 if group=='real_unique' else r['label']==1 and (group=='fake_all8' or r['domain']!='SD1.4'))]
        report['groups'][group]={}
        for a in ARMS:
            chosen=[r for r in selected if r['arm']==a]
            report['groups'][group][a]={key:stats([r[key] for r in chosen]) for key in keys}
        report['groups'][group]['paired_ci95']=macro_bootstrap(selected,ds,keys,args.bootstrap,args.seed+901+len(ds))
    report['per_source_fake']={}
    for d in DOMAINS:
        group=[r for r in records if r['label']==1 and r['domain']==d]
        report['per_source_fake'][d]={a:{k:stats([r[k] for r in group if r['arm']==a]) for k in keys} for a in ARMS}
        for a in ARMS:
            vals=report['per_source_fake'][d][a]
            source_rows.append({'domain':d,'arm':a,'n':vals['detail4_percentile_delta']['n'],
                'detail4_mean':vals['detail4_percentile_delta']['mean'],
                'detail4_negative_rate':vals['detail4_percentile_delta']['negative_rate'],
                'legacy5_mean':vals['legacy5_percentile_delta']['mean'],
                **{m+'_pct_delta':vals[m+'_percentile_delta']['mean'] for m in METRICS5}})
    write_rows(out/'per_source_detail4.csv',source_rows)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n========== PRIMARY RESULTS (Detail4 percentile contrast) ==========',flush=True)
    for grp in main_groups:
        for a in ARMS:
            v=report['groups'][grp][a]['detail4_percentile_delta']
            ci=report['groups'][grp]['paired_ci95']['detail4_percentile_delta'][a]
            print(grp,a,'n=',v['n'],'mean=',f'{v["mean"]:+.6f}',
                  '95%CI=',[round(ci['ci95_low'],6),round(ci['ci95_high'],6)],flush=True)
        diff=report['groups'][grp]['paired_ci95']['detail4_percentile_delta']['delta_Debias_minus_BCE']
        print(grp,'paired delta Debias-BCE:',diff,flush=True)
    print('DETAIL4 CHARACTERIZATION: PASS',flush=True)
    print('SAVED:',out,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/EviRoute_edge'))
    ap.add_argument('--out',type=Path,default=Path('evaluation/early9_detail4_characterization_240_v1'))
    ap.add_argument('--stage',choices=('all','extract','analyze'),default='all')
    ap.add_argument('--fake-per-source',type=int,default=30,help='1..30, nested subset of frozen faithfulness240 fake images')
    ap.add_argument('--real-count',type=int,default=240,help='1..500 unique shared real images')
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--seed',type=int,default=20261009)
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--max-near-ties',type=int,default=2)
    ap.add_argument('--max-boundary-score-gap',type=float,default=0.0005)
    args=ap.parse_args()
    if not (1<=args.fake_per_source<=30 and 1<=args.real_count<=500 and args.batch_size>=1 and
            args.workers>=0 and args.bootstrap>=100 and args.max_near_ties>=0 and args.max_boundary_score_gap>0):
        ap.error('Invalid Detail4 arguments')
    root=args.root.resolve();out=(root/args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    if args.stage in ('all','extract'):do_extract(args,root,out)
    if args.stage in ('all','analyze'):do_analyze(args,root,out)


if __name__=='__main__':main()
