"""Frozen identity, data protocol and numerical auditing for Early-9 deployment.

This helper never writes into existing P0 export paths.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path('/root/autodl-tmp/EviRoute_edge')
PAIR = Path('evaluation/early9_RBCE_vs_RDebias_genimage8000_v1')
MANIFEST = Path('evaluation/genimage_8domain_eval8000/manifest.csv')
OUTPUT_BASE = Path('export/early9_pair_deployment_v1')
CHECKPOINTS = {
    'R_BCE': ('distillation/p0_matched_pair_20k_v1/R_BCE/final_step.pt',
              '3258e32522147a5bb434315e2cb4070f5e348dac65f44b26521f41d94933ebe2'),
    'R_Debias': ('distillation/p0_matched_pair_20k_v1/R_Debias/final_step.pt',
                 'ddfe015a113d137843427a2f547216b7ee1d84eb3cae28717269b4789e1c0a90'),
}
FROZEN_NPZ_SHA = '5bc635bd70b4f8d47ca82bed686639a36857272079bbc769467d3fa8c773d9da'
FROZEN_MANIFEST_SHA = '4d3f05aed0a8d94b4bfe79ddc10e1fc34636e2f18df19f58ba6566b8d5c81a3e'
FROZEN_LOADER_SHA = 'c04f805b1315b04bd81aa5c199eea3e9435a3a7bf88da0dbd2329aa12ac2c07f'
DOMAINS = ('ADM','BigGAN','Midjourney','VQDM','GLIDE','SD1.4','SD1.5','Wukong')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validated_files(root, arm):
    root = Path(root).resolve()
    if arm not in CHECKPOINTS:
        raise ValueError('Unknown arm: '+str(arm))
    c_rel, c_hash = CHECKPOINTS[arm]
    files = {
        'checkpoint': (root / c_rel, c_hash),
        'npz': (root / PAIR / 'paired_outputs.npz', FROZEN_NPZ_SHA),
        'manifest': (root / MANIFEST, FROZEN_MANIFEST_SHA),
        'loader': (root / 'evaluation/eval_early9_p0.py', FROZEN_LOADER_SHA),
    }
    for title, (path, expected) in files.items():
        if not path.is_file():
            raise FileNotFoundError(f'{title} missing: {path}')
        got = sha256(path)
        if got.lower() != expected.lower():
            raise RuntimeError(f'{title} SHA256 unexpected; STOP: {got} != {expected}; path={path}')
    for name in ('export/export_eviroute_early9_p0_onnx.py',
                 'export/experiment_early9_external_patchify_onnx.py',
                 'export/validate_early7_deployment.py',
                 'models/eviroute.py',
                 'distillation/train_early9_lora.py'):
        if not (root/name).is_file():
            raise FileNotFoundError(f'Original verified exporter dependency missing: {root/name}')
    return {key:str(value[0]) for key,value in files.items()}


def pack_rgb_to_nchw(x):
    """Exact original external Patchify: RGB float32 NCHW to 1x768x1x196."""
    x = np.asarray(x)
    if x.shape != (1,3,224,224) or x.dtype != np.float32:
        raise ValueError(f'RGB tensor should be (1,3,224,224), float32, got {x.shape}, {x.dtype}')
    return np.ascontiguousarray(x.reshape(1,3,14,16,14,16)
                                  .transpose(0,1,3,5,2,4).reshape(1,768,1,196))


def topk_logit(x):
    x = np.asarray(x).astype(np.float64).reshape(-1)
    if x.size != 196 or not np.isfinite(x).all():
        raise ValueError('196 finite patch scores required')
    return float(np.sort(x)[-10:].mean())


def stats(ref, obs):
    a, b = np.asarray(ref, dtype=np.float64).reshape(-1), np.asarray(obs, dtype=np.float64).reshape(-1)
    if a.size!=196 or b.size!=196 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Bad 196-dimensional patch scores')
    den = np.linalg.norm(a)*np.linalg.norm(b)
    if den == 0: raise ValueError('Zero-norm patch score vector')
    ia = set(np.argsort(-a)[:10].tolist()); ib=set(np.argsort(-b)[:10].tolist())
    return {
        'cosine':float(np.dot(a,b)/den),
        'mae':float(np.mean(np.abs(a-b))),
        'max_abs':float(np.max(np.abs(a-b))),
        'logit_error':float(abs(topk_logit(a)-topk_logit(b))),
        'top10_overlap':int(len(ia & ib)),
        'prediction_match':bool((topk_logit(a)>=0)==(topk_logit(b)>=0)),
    }


def summarize(rows):
    if not rows: raise ValueError('No test cases')
    return {'cases':len(rows),
            'min_cosine':min(x['cosine'] for x in rows),
            'mean_mae':float(np.mean([x['mae'] for x in rows])),
            'max_abs':max(x['max_abs'] for x in rows),
            'max_logit_error':max(x['logit_error'] for x in rows),
            'mean_top10_overlap':float(np.mean([x['top10_overlap'] for x in rows])),
            'prediction_matches':sum(x['prediction_match'] for x in rows)}


def strict_float_gate(s, label):
    if (s['min_cosine']<.99999 or s['max_abs']>.01 or s['max_logit_error']>.01 or
        s['mean_top10_overlap']<9.5 or s['prediction_matches']!=s['cases']):
        raise RuntimeError(f'{label} did not meet established P0 export gate: {s}')


def frozen_top10_audit(logit, indices, scores, cache_logit, cache_indices,
                       *, max_boundary_gap=.0005):
    """Replicate near-tie policy WITHOUT silently ignoring changed Top10 membership."""
    err=abs(float(logit)-float(cache_logit))
    if err > .002 + .0001*abs(float(cache_logit)):
        raise RuntimeError(f'Frozen paired logit mismatch: abs_error={err}')
    old=np.asarray(cache_indices,dtype=np.int64).reshape(-1)
    new=np.asarray(indices,dtype=np.int64).reshape(-1)
    if old.size !=10 or new.size!=10:
        raise ValueError('Expected two 10-element Top10 lists')
    if len(set(old))!=10 or len(set(new))!=10 or min(old.min(),new.min())<0 or max(old.max(),new.max())>=196:
        raise ValueError('Duplicate/out-of-range patch indices')
    if set(old)==set(new):
        return {'logit_abs_error':err,'top10_set_match':True,'near_tie':False,'top10_order_match':bool(np.array_equal(old,new))}
    old_only=list(set(old)-set(new)); new_only=list(set(new)-set(old))
    if len(old_only)!=1 or len(new_only)!=1 or set(old[:9])!=set(new[:9]):
        raise RuntimeError(f'Top10 substantial membership drift: old_only={old_only} new_only={new_only}')
    s=np.asarray(scores,dtype=np.float64).reshape(-1)
    gap=abs(float(s[old_only[0]]-s[new_only[0]]))
    if gap>max_boundary_gap:
        raise RuntimeError(f'Top10 cutoff drift gap exceeds limit: gap={gap}')
    return {'logit_abs_error':err,'top10_set_match':False,'near_tie':True,
            'gap':gap,'old_only':old_only,'new_only':new_only,
            'top10_order_match':False}


def select_cases(root, arm, count):
    """Predeclared deterministic, balanced subset; 16 cases = 8 unique real + 8 fake (all sources)."""
    import pandas as pd
    if count!=16:
        raise ValueError('V1 protocol fixes --num-check=16 to cover all eight generators')
    df=pd.read_csv(Path(root)/MANIFEST).sort_values('global_index')
    if len(df)!=8000 or sorted(df['global_index'].tolist())!=list(range(8000)):
        raise RuntimeError('Manifest index integrity mismatch')
    if set(df['domain'])!=set(DOMAINS):raise RuntimeError('Unexpected generation sources')
    rng=np.random.default_rng(20261010)
    pool_real=df[(df['domain']=='ADM') & (df['label']==0)]
    real_indices=rng.choice(pool_real.index.to_numpy(),size=8,replace=False).tolist()
    fake_indices=[]
    for domain in DOMAINS:
        pool=df[(df['domain']==domain)&(df['label']==1)]
        if len(pool)!=500:raise RuntimeError(f'Unexpected {domain} fake count')
        fake_indices.append(int(rng.choice(pool.index.to_numpy())))
    selected=df.iloc[real_indices+fake_indices]
    if len(selected)!=16 or len(set(selected['path']))!=16:
        raise RuntimeError('Export cases not unique')
    with np.load(Path(root)/PAIR/'paired_outputs.npz',allow_pickle=False) as z:
        gid=z['global_index'].astype(np.int64)
        if gid.shape!=(8000,) or set(gid.tolist())!=set(range(8000)):
            raise RuntimeError('Frozen outputs global_index integrity mismatch')
        loc={int(k):i for i,k in enumerate(gid)}
        cache_logit=z[arm+'_logits']; cache_idx=z[arm+'_top10_indices']
        if cache_logit.shape!=(8000,) or cache_idx.shape!=(8000,10):
            raise RuntimeError('Frozen outputs shape mismatch')
        rows=[]
        for row in selected.itertuples(index=False):
            j=loc[int(row.global_index)]
            if not Path(row.path).is_file():raise FileNotFoundError(row.path)
            rows.append({'global_index':int(row.global_index),'domain':str(row.domain),
                         'label':int(row.label),'path':str(row.path),
                         'frozen_logit':float(cache_logit[j]),
                         'frozen_top10':cache_idx[j].astype(int).tolist()})
    return rows


def save_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
