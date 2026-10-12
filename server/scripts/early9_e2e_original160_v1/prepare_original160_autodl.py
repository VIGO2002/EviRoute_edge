#!/usr/bin/env python3
"""Create a deterministic original JPEG/PNG corpus from the existing audited Eval480 manifest.

This is an EXPERIMENTAL ENGINEERING BENCHMARK, not a new held-out generalization set.
Never converts the input files to 224x224; copies their ORIGINAL encoded bytes.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ALLOWED = {'.jpg', '.jpeg', '.png'}
EXPECTED = ('ADM','BigGAN','Midjourney','VQDM','GLIDE','SD1.4','SD1.5','Wukong')

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    return h.hexdigest()

def selection(manifest:dict, per_class:int, seed:int) -> list[dict]:
    names=('global_indices','labels','domains','source_paths')
    if not all(isinstance(manifest.get(k),list) and len(manifest[k])==480 for k in names):
        raise ValueError('Eval480 manifest missing 480 correctly sized provenance vectors')
    if manifest.get('n_images')!=480 or manifest.get('status')!='prepared':
        raise ValueError('Must use the previously audited prepared Eval480 manifest')
    if len(set(map(int,manifest['global_indices'])))!=480:
        raise ValueError('Global indices not unique')
    if len(set(manifest['source_paths']))!=480:
        raise ValueError('Original source paths not unique')
    if per_class != 10: raise ValueError('Protocol v1 is fixed at exactly 10 images per source and class (160 total)')
    rows=[dict(global_index=int(manifest['global_indices'][i]), label=int(manifest['labels'][i]),
               domain=str(manifest['domains'][i]), original_path=str(manifest['source_paths'][i])) for i in range(480)]
    c=Counter((r['domain'],r['label']) for r in rows)
    if set(c)!={(d,l) for d in EXPECTED for l in (0,1)} or any(v!=30 for v in c.values()):
        raise ValueError('Need exactly 30 Real and 30 Fake for each of eight domains')
    rng=np.random.default_rng(seed)
    chosen=[]
    for domain in EXPECTED:
        for label in (0,1):
            candidates=sorted((r for r in rows if r['domain']==domain and r['label']==label),
                              key=lambda r:r['global_index'])
            positions=sorted(int(x) for x in rng.choice(len(candidates),per_class,replace=False))
            chosen.extend(candidates[p] for p in positions)
    return chosen

def prepare(manifest_path:Path, root:Path, per_class:int=10,seed:int=20261010) -> tuple[Path,Path]:
    manifest_path=manifest_path.expanduser().resolve()
    root=root.expanduser().resolve()
    m=json.loads(manifest_path.read_text(encoding='utf-8'))
    rows=selection(m,per_class,seed)
    rootdir=root/'full_pipeline_original160_v1'
    archive=root/'full_pipeline_original160_v1.tar.gz'
    if rootdir.exists() or archive.exists():
        raise FileExistsError('Refusing to overwrite existing corpus / archive: '+str(rootdir))
    # Preflight all sources BEFORE writing anything.
    for r in rows:
        p=Path(r['original_path'])
        if not p.is_file() or p.suffix.lower() not in ALLOWED:
            raise FileNotFoundError(f'Original JPEG/PNG not available: {p}')
        with Image.open(p) as img:
            if img.width<1 or img.height<1:
                raise ValueError(f'Invalid dimensions {p}')
    rootdir.mkdir(parents=True)
    out_rows=[]
    try:
        for i,r in enumerate(rows):
            src=Path(r['original_path'])
            klass='fake' if r['label'] else 'real'
            dest_name=f'{r["global_index"]:04d}_{src.stem[:48]}{src.suffix.lower()}'
            rel=Path(r['domain'])/klass/dest_name
            dst=rootdir/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dst)
            if sha256(src)!=sha256(dst):
                raise RuntimeError('Copy checksum mismatch: '+str(src))
            with Image.open(dst) as im:
                width,height=im.size
            out_rows.append({'relative_path':rel.as_posix(), 'label':r['label'], 'domain':r['domain'],
                             'global_index':r['global_index'],'source_path':r['original_path'],
                             'original_width':width,'original_height':height,
                             'input_sha256':sha256(dst),'size_bytes':dst.stat().st_size})
            if (i+1)%40==0:
                print(f'Original images copied & verified: {i+1}/{len(rows)}',flush=True)
        with (rootdir/'labels.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=('relative_path','label'))
            w.writeheader()
            for r in out_rows: w.writerow({'relative_path':r['relative_path'],'label':r['label']})
        meta={'status':'prepared','purpose':'original-file end-to-end inference benchmarking; NOT a pristine holdout',
              'n_images':len(out_rows),'source_manifest_sha256':sha256(manifest_path),
              'selection_seed':seed,'selected_per_source_and_label':per_class,
              'selection':'fixed random without replacement within each 30-image source+label stratum',
              'original_encoded_bytes_copied_unchanged':True,'expected_models':['R_BCE','R_Debias'],
              'images':out_rows}
        (rootdir/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(rootdir,arcname=rootdir.name)
        print('ORIGINAL160 PREPARE: PASS')
        print('Original encoded image files:',len(out_rows))
        print('Archive:',archive,'MiB=',round(archive.stat().st_size/(1024**2),2))
        print('Archive SHA256:',sha256(archive))
        return rootdir,archive
    except Exception:
        # Never leave an apparently valid incomplete directory.
        shutil.rmtree(rootdir,ignore_errors=True)
        if archive.exists():archive.unlink()
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=Path('evaluation/early9_pair_480_v1/manifest.json'))
    p.add_argument('--out-parent',type=Path,default=Path('evaluation/early9_e2e_original160_v1'))
    p.add_argument('--per-class',type=int,default=10)
    p.add_argument('--seed',type=int,default=20261010)
    args=p.parse_args()
    prepare(args.manifest,args.out_parent,args.per_class,args.seed)
if __name__=='__main__': main()
