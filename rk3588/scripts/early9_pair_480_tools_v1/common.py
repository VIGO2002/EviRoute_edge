"""Shared exact data protocol for Early-9 R-BCE / R-Debias 480-image board audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS = ('R_BCE', 'R_Debias')
DOMAINS = ('ADM', 'BigGAN', 'Midjourney', 'VQDM', 'GLIDE', 'SD1.4', 'SD1.5', 'Wukong')
MODEL_SHA256 = {
    'R_BCE': '808cecbf53a60d3071d622a32792f7cce782a22115c97b2ead9d7cbda75cf597',
    'R_Debias': '40db8e07ef6fe537c913d19bdbbb0fe01ec0c1f3a1fba14d1e86a6ae6e4a251c',
}
CKPT_SHA256 = {
    'R_BCE': '3258e32522147a5bb434315e2cb4070f5e348dac65f44b26521f41d94933ebe2',
    'R_Debias': 'ddfe015a113d137843427a2f547216b7ee1d84eb3cae28717269b4789e1c0a90',
}
PAIRED_SHA256 = '5bc635bd70b4f8d47ca82bed686639a36857272079bbc769467d3fa8c773d9da'
MANIFEST_SHA256 = '4d3f05aed0a8d94b4bfe79ddc10e1fc34636e2f18df19f58ba6566b8d5c81a3e'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sha(path, expected, what):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {what}: {path}')
    got = sha256(path)
    if got.lower() != expected.lower():
        raise RuntimeError(f'{what} SHA256 MISMATCH: {got} != {expected}: {path}')
    return got


def top10(scores):
    a = np.asarray(scores, dtype=np.float64).reshape(-1)
    if a.size != 196 or not np.isfinite(a).all():
        raise ValueError('Need 196 finite patch scores')
    # Stable ranks are necessary only in the event of numerically equal scores.
    ids = np.argsort(-a, kind='stable')[:10]
    return float(a[ids].mean()), ids.astype(np.int16)


def pack_rgb(x):
    """Byte-exact with validated P0/R-BCE/R-Debias external CPU patchify."""
    if x.shape != (1, 3, 224, 224) or x.dtype != np.float32:
        raise ValueError(f'Expected (1,3,224,224) float32, received {x.shape}/{x.dtype}')
    return np.ascontiguousarray(x.reshape(1,3,14,16,14,16)
                                  .transpose(0,1,3,5,2,4).reshape(1,768,1,196))


def rgb_uint8_to_packed(rgb):
    """Exact original preprocessing after Pillow RGB/BICUBIC resize to 224x224."""
    rgb = np.asarray(rgb)
    if rgb.shape != (224,224,3) or rgb.dtype != np.uint8:
        raise ValueError(f'Bad uint8 RGB shape/dtype {rgb.shape}/{rgb.dtype}')
    nchw = np.ascontiguousarray((rgb.astype(np.float32)/np.float32(255.0)).transpose(2,0,1)[None])
    return pack_rgb(nchw)


def region_counts(indices):
    a=np.asarray(indices, dtype=np.int64).reshape(-1)
    if a.size!=10 or (a<0).any() or (a>=196).any():
        raise ValueError('Bad Top10 ids')
    r=a//14; c=a%14
    b1=(r==0)|(r==13)|(c==0)|(c==13)
    b2=(r<2)|(r>11)|(c<2)|(c>11)
    return int(b1.sum()), int((b2&~b1).sum()), int((~b2).sum())


def json_save(path, obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n',encoding='utf-8')


def read_meta_inputs(dir_path):
    """Check meta and arrays all map to same global_index and are finite."""
    root=Path(dir_path)
    meta=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    arrays=np.load(root/'references.npz',allow_pickle=False)
    ins=np.load(root/'inputs_224_uint8.npz',allow_pickle=False)
    gids=arrays['global_index']
    labels=arrays['labels']
    domains=arrays['domain']
    rgb=ins['rgb']
    if gids.shape!=(480,) or len(set(gids.tolist()))!=480 or labels.shape!=(480,):
        raise RuntimeError('Invalid 480 identity arrays')
    if rgb.shape!=(480,224,224,3) or rgb.dtype!=np.uint8:
        raise RuntimeError(f'Unexpected RGB uint8 array: {rgb.shape}/{rgb.dtype}')
    if not np.array_equal(ins['global_index'],gids):
        raise RuntimeError('Input/global_index mismatch')
    if not np.array_equal(meta['global_indices'],gids):
        raise RuntimeError('Manifest/global_index mismatch')
    if not np.array_equal(meta['labels'],labels):
        raise RuntimeError('Manifest/label mismatch')
    if not np.array_equal(meta['domains'],domains):
        raise RuntimeError('Manifest/domain mismatch')
    if len(np.unique(labels,return_counts=True)[1])!=2 or np.count_nonzero(labels==0)!=240 or np.count_nonzero(labels==1)!=240:
        raise RuntimeError('Expected 240 real + 240 fake')
    if set(domains.tolist())!=set(DOMAINS):
        raise RuntimeError('Bad domain list')
    for arm in ARMS:
        s=arrays[arm+'_scores']
        z=arrays[arm+'_logits']
        ix=arrays[arm+'_top10']
        if s.shape!=(480,196) or s.dtype!=np.float32 or z.shape!=(480,) or ix.shape!=(480,10):
            raise RuntimeError(f'Bad {arm} reference arrays')
        if not np.isfinite(s).all() or not np.isfinite(z).all():
            raise RuntimeError('Non-finite reference arrays')
        derived=np.mean(np.take_along_axis(s,ix.astype(np.int64),axis=1),axis=1)
        if not np.allclose(derived,z,atol=.002,rtol=1e-4):
            raise RuntimeError(f'Logit does not match selected Top10 for {arm}')
    for dom in DOMAINS:
        if np.count_nonzero((domains==dom)&(labels==1))!=30 or np.count_nonzero((domains==dom)&(labels==0))!=30:
            raise RuntimeError(f'Not balanced 30 real/30 fake for {dom}')
    return meta,arrays,ins
