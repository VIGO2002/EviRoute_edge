#!/usr/bin/env python3
"""AutoDL CPU only: package the exact 480-image Detail4 cohort and audited patch-score cache."""
from __future__ import annotations
import argparse
import csv
import json
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from common import (ARMS, DOMAINS, CKPT_SHA256, MODEL_SHA256, PAIRED_SHA256,
                    MANIFEST_SHA256, sha256, verify_sha, json_save, top10, region_counts)

DETAIL_REL=Path('evaluation/early9_detail4_240fake_240real_v1')
EVAL_REL=Path('evaluation/genimage_8domain_eval8000/manifest.csv')
PAIR_REL=Path('evaluation/early9_RBCE_vs_RDebias_genimage8000_v1/paired_outputs.npz')


def load_rows(path):
    with Path(path).open('r',encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))


def make_cohort(selected, cached_gids, cached_labels, *, real_partition_seed):
    """Preserve source detail4 CSV order, assign each REAL image once to an eval source."""
    if len(selected)!=480:
        raise RuntimeError(f'Selection manifest expected 480 rows, found {len(selected)}')
    if len(set(x['global_index'] for x in selected))!=480:
        raise RuntimeError('Duplicate global_index in selection manifest')
    if [int(x['global_index']) for x in selected]!=cached_gids.astype(int).tolist():
        raise RuntimeError('Selection manifest / Detail4 cache row-order mismatch')
    if [int(x['label']) for x in selected]!=cached_labels.astype(int).tolist():
        raise RuntimeError('Selection manifest / Detail4 cache labels mismatch')
    real=np.flatnonzero(cached_labels==0)
    fake=np.flatnonzero(cached_labels==1)
    if real.size!=240 or fake.size!=240:
        raise RuntimeError('Need 240 unique Real, 240 Fake')
    for d in DOMAINS:
        if sum(x['domain']==d and int(x['label'])==1 for x in selected)!=30:
            raise RuntimeError('Wrong fake domain distribution')
    rng=np.random.default_rng(real_partition_seed)
    perm=rng.permutation(real)
    assigned={int(row): DOMAINS[i//30] for i,row in enumerate(perm)}
    result=[]
    for i,r in enumerate(selected):
        result.append({'global_index':int(r['global_index']),
                       'label':int(r['label']),
                       'domain': r['domain'] if int(r['label'])==1 else assigned[i],
                       'original_domain':r['domain'],
                       'path':r['path']})
    counts=Counter((r['domain'],r['label']) for r in result)
    for dom in DOMAINS:
        if counts[(dom,0)]!=30 or counts[(dom,1)]!=30:
            raise RuntimeError('Unbalanced real/fake assignment')
    if len({r['path'] for r in result})!=480:
        raise RuntimeError('480 images not physically unique')
    return result


def prepare(args):
    root=args.root.resolve(); src=root/DETAIL_REL
    manifest=root/EVAL_REL;pair=root/PAIR_REL
    verify_sha(manifest,MANIFEST_SHA256,'GenImage Eval8000 manifest')
    verify_sha(pair,PAIRED_SHA256,'strict paired outputs')
    metap=src/'patch_scores_meta.json'
    meta=json.loads(metap.read_text(encoding='utf-8'))
    if meta.get('status')!='extracted' or meta.get('n_fake')!=240 or meta.get('n_real_unique')!=240:
        raise RuntimeError('Detail4 full cache has incorrect status/size')
    prov=meta['provenance']
    if (prov['paired_npz_sha256']!=PAIRED_SHA256 or prov['manifest_sha256']!=MANIFEST_SHA256 or
            any(prov['checkpoints'][a]!=CKPT_SHA256[a] for a in ARMS)):
        raise RuntimeError('Detail4 patch scores are NOT from the frozen 2625 paired models')
    # Complete extraction audits are 100% aligned on this exact 480 cohort.
    for arm in ARMS:
        for subset in ('fake','real'):
            if meta['audits'][arm][subset]['n']!=240:
                raise RuntimeError('Incomplete Detail4 original numerical audit: '+arm+' '+subset)
        if meta['audits'][arm]['fake']['top10_set_match_count']!=240 or meta['audits'][arm]['fake']['allowed_near_tie_count']!=0:
            raise RuntimeError('Detail4 fake Top10 cache not exactly aligned')
        if meta['audits'][arm]['real']['fake_score_top10_set_match_count']!=240:
            raise RuntimeError('Detail4 real Top10 cache not exactly aligned')
    cached=np.load(src/'patch_scores.npz',allow_pickle=False)
    ids=cached['global_index'];labels=cached['label']
    rows=make_cohort(load_rows(src/'selection_manifest.csv'), ids, labels, real_partition_seed=args.partition_seed)
    original=load_rows(manifest)
    mapping={int(r['global_index']):r for r in original}
    if len(mapping)!=8000:
        raise RuntimeError('Original eval manifest missing global IDs')
    for r in rows:
        srcrow=mapping[r['global_index']]
        if (srcrow['path']!=r['path'] or int(srcrow['label'])!=r['label'] or
            srcrow['domain']!=r['original_domain']):
            raise RuntimeError('Original manifest mismatch at '+str(r['global_index']))
    n=len(rows)
    arrays={'global_index': ids.astype(np.int64), 'labels':labels.astype(np.int8),
            'domain':np.asarray([x['domain'] for x in rows],dtype='<U16')}
    with np.load(pair,allow_pickle=False) as frozen:
        positions={int(gid):i for i,gid in enumerate(frozen['global_index'].tolist())}
        for arm in ARMS:
            score=cached[arm+'_scores'].astype(np.float32)
            z=cached[arm+'_logits'].astype(np.float32)
            ix=cached[arm+'_current_fake_top10'].astype(np.int16)
            if score.shape!=(480,196) or ix.shape!=(480,10) or z.shape!=(480,):
                raise RuntimeError('Invalid patch score cache '+arm)
            for i,r in enumerate(rows):
                f=positions[r['global_index']]
                if abs(float(z[i])-float(frozen[arm+'_logits'][f]))>(.002+.0001*abs(float(frozen[arm+'_logits'][f]))):
                    raise RuntimeError(f'Frozen logit mismatch {arm}: {r["global_index"]}')
                if set(ix[i].tolist())!=set(frozen[arm+'_top10_indices'][f].tolist()):
                    raise RuntimeError(f'Frozen Top10 membership mismatch {arm}: {r["global_index"]}')
                reconstructed,derived=top10(score[i])
                if abs(reconstructed-float(z[i]))>.002:
                    raise RuntimeError(f'Wrong cached Top10 aggregate: {arm}/{r["global_index"]}')
                if set(derived.tolist())!=set(ix[i].tolist()):
                    raise RuntimeError(f'Cached scores choose different Top10: {arm}/{r["global_index"]}')
            arrays[arm+'_scores']=score
            arrays[arm+'_logits']=z
            arrays[arm+'_top10']=ix
    out=args.out.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    out.mkdir(parents=True)
    imgs=np.empty((n,224,224,3),dtype=np.uint8)
    bicubic=Image.Resampling.BICUBIC if hasattr(Image,'Resampling') else Image.BICUBIC
    for i,r in enumerate(rows):
        p=Path(r['path'])
        if not p.is_file():
            raise FileNotFoundError(f'Image missing at {r["global_index"]}: {p}')
        with Image.open(p) as im:
            im=im.convert('RGB').resize((224,224),resample=bicubic)
            imgs[i]=np.asarray(im,dtype=np.uint8)
        if (i+1)%60==0:
            print(f'RGB uint8 resize {i+1}/{n}',flush=True)
    np.savez_compressed(out/'inputs_224_uint8.npz',global_index=ids.astype(np.int64),rgb=imgs)
    np.savez_compressed(out/'references.npz',**arrays)
    report={
        'status':'prepared','protocol':'Early9 paired Eval480; EXACT Detail4 images and cached PyTorch scores',
        'caution':'EXPLORATORY, previously inspected Eval8000 subset; not independent final holdout',
        'global_indices':ids.astype(int).tolist(),'labels':labels.astype(int).tolist(),
        'domains':[r['domain'] for r in rows],
        'original_domains':[r['original_domain'] for r in rows],
        'real_partition':'240 unique shared real images randomly partitioned 30 per source; groups used only for paired source-balanced AUC',
        'real_partition_seed':args.partition_seed,
        'fake_per_domain':30,'real_per_domain':30,'n_images':480,
        'source_paths':[r['path'] for r in rows],
        'provenance':{'paired_npz_sha256':sha256(pair),'eval_manifest_sha256':sha256(manifest),
            'detail4_patch_cache_sha256':sha256(src/'patch_scores.npz'),
            'detail4_cache_metadata_sha256':sha256(metap),
            'detail4_selection_manifest_sha256':sha256(src/'selection_manifest.csv'),
            'checkpoints':CKPT_SHA256,'rk3588_models':MODEL_SHA256},
        'normalization':'PIL RGB bicubic resize 224x224 -> float32(rgb_uint8)/255 -> CHW -> external patchify [1,768,1,196]',
    }
    json_save(out/'manifest.json',report)
    checks={f:sha256(out/f) for f in ('manifest.json','references.npz','inputs_224_uint8.npz')}
    json_save(out/'transport_sha256.json',checks)
    package=out.parent/(out.name+'.tar.gz')
    if package.exists():raise FileExistsError(f'Refusing overwrite existing archive: {package}')
    with tarfile.open(package,'w:gz') as tf:
        for filename in ('manifest.json','references.npz','inputs_224_uint8.npz','transport_sha256.json'):
            tf.add(out/filename,arcname='early9_pair_480_v1/'+filename)
    print('\nPREPARE EVAL480: PASS')
    print('Data package:',package,'size MiB:',round(package.stat().st_size/1024**2,2))
    print('Reference SHA256:',sha256(out/'references.npz'))
    print('Same 480 images for both models; 30 balanced real+fake per source; no training/GPU.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/EviRoute_edge'))
    p.add_argument('--out',type=Path,default=Path('/root/autodl-tmp/EviRoute_edge/evaluation/early9_pair_480_v1'))
    p.add_argument('--partition-seed',type=int,default=20261010)
    args=p.parse_args()
    prepare(args)

if __name__=='__main__':main()
