#!/usr/bin/env python3
"""R-BCE / R-Debias: paired original P0 20k-data, 2625-update recipe.

Based on the verified historical train_early9_lora_data_v2.py. It deliberately
preserves the historical head-dropout eval-mode state after validation.
This script NEVER edits source P0 or previous experimental checkpoints.
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from early9_p0_matched_pair_core import CFG, compute_loss, fake_mass_stats, merge_mass, masks, improved

ROOT_DEFAULT = Path('/root/autodl-tmp/EviRoute_edge')
SOURCE_HASHES = {
    'train_early9_lora.py': 'e31e0b7154d99efa4de2d1ca723b50940dd697e56590f627d21cda145f4dc9ba',
    'train_early9_lora_data_v2.py': 'e417dd8c6be9c792c8d802bbe1e9eb2c8678f43a175d56ebac663d843841760b',
}
P0_SHA = '79f1983e83d905e0e26f2f64ec181f1800dcff80d399e31341c29c52a0a2e995'
INIT_SCHEMA = 'early9_matched_p0_20k_shared_init_seed42_v1'
CKPT_SCHEMA = 'early9_matched_p0_20k_pair_v1'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def labeled_split_sha(paths, labels):
    h=hashlib.sha256()
    for path,label in zip(paths,labels):
        h.update((str(Path(str(path)).resolve())+'\t'+str(int(label))+'\n').encode())
    return h.hexdigest()


def as_abs(root,path):
    return path if path.is_absolute() else root/path


def verify_old_files(root):
    for name,expected in SOURCE_HASHES.items():
        p=root/'distillation'/name
        if not p.is_file():
            raise FileNotFoundError(p)
        actual=sha256(p)
        if actual!=expected:
            raise RuntimeError(f'Historical source changed: {name}: {actual} != {expected}')
    print('HISTORICAL SOURCE SHA256: PASS',flush=True)


def verify_historical_p0_recipe(root,args):
    """Ensure CLI really matches historical *executed* training_args, not defaults."""
    ckpt_path=root/'distillation/lora_runs/P0_Early9_20k_classic/best.pt'
    if not ckpt_path.is_file():
        raise FileNotFoundError('Historical P0 checkpoint needed for recipe audit only: '+str(ckpt_path))
    actual_sha=sha256(ckpt_path)
    if actual_sha != P0_SHA:
        raise RuntimeError(f'Historical P0 SHA mismatch: {actual_sha}')
    ckpt=torch.load(ckpt_path,map_location='cpu',weights_only=False)
    meta=ckpt.get('training_args')
    if not isinstance(meta,dict):
        raise RuntimeError('Historical P0 lacks training_args dictionary')
    for recorded, current in [('max_steps',args.max_steps),('val_every',args.val_every),
                               ('batch_size',args.batch_size),('workers',args.workers),
                               ('seed',args.seed),('rank',args.rank)]:
        if recorded not in meta:
            raise RuntimeError('Historical P0 did not record '+recorded+'; verify original log before training')
        if int(meta[recorded]) != int(current):
            raise RuntimeError(f'Historical P0 {recorded}={meta[recorded]}, current={current}')
    for recorded,current in [('lr',args.lr),('weight_decay',args.weight_decay),('alpha',args.alpha)]:
        if recorded not in meta or abs(float(meta[recorded])-float(current))>1e-12:
            raise RuntimeError(f'Historical P0 {recorded} mismatch: {meta.get(recorded)} vs {current}')
    if meta.get('augmentation')!=args.augmentation:
        raise RuntimeError('Historical P0 augmentation mismatch')
    old_manifest=Path(str(meta.get('train_manifest','')))
    if not old_manifest.is_absolute():old_manifest=root/old_manifest
    if old_manifest.resolve()!=args.train_manifest.resolve():
        raise RuntimeError(f'Historical P0 manifest mismatch: {old_manifest} vs {args.train_manifest}')
    print('HISTORICAL P0 RECIPE: PASS',json.dumps({k:meta[k] for k in
        ('max_steps','val_every','batch_size','workers','seed','rank','lr','weight_decay','alpha','augmentation')}),flush=True)
    return actual_sha


def model_and_data(root,args):
    # Explicitly use the OLD trainer and dataset construction, not a recreated architecture.
    sys.path.insert(0,str(root/'distillation'))
    import train_early9_lora as old
    import train_early9_lora_data_v2 as v2
    if root.resolve() != old.ROOT.resolve():
        raise RuntimeError('Original trainer hardcodes ROOT; use /root/autodl-tmp/EviRoute_edge')
    if args.batch_size != 16 or args.seed != 42 or args.lr != 1e-4 or args.weight_decay != 1e-4:
        raise RuntimeError('Matched P0 recipe locks seed=42, batch16, LR1e-4, WD1e-4')
    if args.max_steps != 2625 or args.val_every != 175:
        raise RuntimeError('Matched P0 recipe locks max_steps=2625, val_every=175')
    if args.augmentation != 'classic' or args.rank!=8 or args.alpha!=8.: 
        raise RuntimeError('Expected classic augmentation and rank/alpha 8/8')
    eval_df=pd.read_csv(old.EVAL_MANIFEST)
    eval_paths={v2.canonical(p) for p in eval_df['path']}
    original_train,train_cache=old.load_data('train',eval_paths)
    val_set,_=old.load_data('val',eval_paths)
    train_paths,train_labels=v2.prepare_training_set(args.train_manifest,
        original_train,val_set,eval_paths)
    train_set=v2.TrainingImages(train_paths,train_labels,args.augmentation)
    baseline=torch.load(old.BASE_CKPT,map_location='cpu',weights_only=False)
    model=old.Early9LoRA(baseline,rank=args.rank,alpha=args.alpha).to('cuda:0')
    old.check_features(model,original_train,train_cache,torch.device('cuda:0'))
    return old,v2,model,train_set,val_set


def signature(root,args,old,train_set,val_set):
    return {
        'base_checkpoint_sha256':sha256(old.BASE_CKPT),
        'train_manifest_sha256':sha256(args.train_manifest),
        'train_split_ordered_sha256':labeled_split_sha(train_set.paths,train_set.labels),
        'val_split_ordered_sha256':labeled_split_sha(val_set.paths,val_set.labels),
        'eval_manifest_sha256':sha256(old.EVAL_MANIFEST),
        'source_sha256':SOURCE_HASHES,
        'train_count':len(train_set), 'val_count':len(val_set),
    }


def lora_dict(model):
    return model.lora_state()


def head_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}


def fingerprint_weights(model):
    h=hashlib.sha256()
    for prefix,values in (('head',head_dict(model)),('lora',lora_dict(model))):
        for name in sorted(values):
            h.update((prefix+':'+name).encode())
            value=values[name].detach().contiguous().numpy()
            h.update(value.tobytes())
    return h.hexdigest()


def pack_init(model,sig):
    return {'schema':INIT_SCHEMA, 'seed':42,'signature':sig,
            'head_state_dict':head_dict(model),'lora_state_dict':lora_dict(model),
            'mu':model.mu.detach().cpu().clone(), 'std':model.std.detach().cpu().clone(),
            'initial_weights_sha256':fingerprint_weights(model)}


def load_shared(model,shared,expected_signature):
    if shared.get('schema')!=INIT_SCHEMA or shared.get('seed')!=42:
        raise RuntimeError('Wrong shared init schema or seed')
    if shared.get('signature')!=expected_signature:
        raise RuntimeError('Shared initialization data/base/source SHA mismatch')
    model.head.load_state_dict(shared['head_state_dict'],strict=True)
    reference=model.lora_state()
    if set(reference)!=set(shared['lora_state_dict']) or len(reference)!=12:
        raise RuntimeError('Expected 12 LoRA A/B tensors in shared init')
    with torch.no_grad():
        for block in (6,7,8):
            for target in ('qkv','proj'):
                linear=getattr(model.backbone.blocks[block].attn,target)
                for part in ('A','B'):
                    k=f'blocks.{block}.attn.{target}.{part}'
                    dest=getattr(linear,part)
                    src=shared['lora_state_dict'][k]
                    if dest.shape!=src.shape:
                        raise RuntimeError(f'LoRA shape mismatch {k}')
                    dest.copy_(src.to(dest.device))
        if not (torch.equal(model.mu.cpu(),shared['mu']) and torch.equal(model.std.cpu(),shared['std'])):
            raise RuntimeError('Feature normalization mismatch')
    if fingerprint_weights(model)!=shared['initial_weights_sha256']:
        raise RuntimeError('Model does not match shared init weights')
    for k,v in shared['lora_state_dict'].items():
        if k.endswith('.B') and int(torch.count_nonzero(v))!=0:
            raise RuntimeError('Pre-P0 LoRA B must be zero initialized')
    print('SHARED INITIAL WEIGHTS: PASS',shared['initial_weights_sha256'],flush=True)


def val_metrics(model,loader):
    """Same logits and sorting as historical evaluate(), add B2/real for diagnostics."""
    model.eval()
    all_y,all_z,all_top=[] , [], []
    with torch.inference_mode():
        for images,labels in loader:
            logits,_,indices=model(images.to('cuda:0',non_blocking=True))
            all_y.append(labels.numpy())
            all_z.append(logits.cpu().numpy())
            all_top.append(indices.cpu().numpy())
    y=np.concatenate(all_y).astype(np.int64)
    z=np.concatenate(all_z)
    top=np.concatenate(all_top)
    b1,b2,_=masks(device='cpu')
    b1,b2=b1.numpy(),b2.numpy()
    result=dict(auc=float(roc_auc_score(y,z)),ap=float(average_precision_score(y,z)),
        acc=float(np.mean((z>=0)==y)),
        bce=float(F.binary_cross_entropy_with_logits(torch.from_numpy(z),torch.from_numpy(y.astype(np.float32))).item()))
    for name,target in [('fake',1),('real',0)]:
        selected=top[y==target]
        result[f'{name}_top1_b1']=float(b1[selected[:,0]].mean())
        result[f'{name}_top10_b1']=float(b1[selected].mean())
        result[f'{name}_top1_b2']=float(b2[selected[:,0]].mean())
        result[f'{name}_top10_b2']=float(b2[selected].mean())
    return result


def checkpoint(model,old,args,arm,epoch,step,metrics,sig,shared_sha,filename):
    # Save using the historical checkpoint function; supplement provenance only.
    old.save_checkpoint(model,args,epoch,metrics,filename)
    ckpt=torch.load(filename,map_location='cpu',weights_only=False)
    ckpt['paired_p0_20k']=dict(schema=CKPT_SCHEMA,arm=arm,step=step,epoch=epoch,
        shared_init_sha256=shared_sha,signature=sig,
        original_train_mode_bug_preserved=True, debias_config=dict(CFG) if arm=='R-Debias' else None,
        selection_rule='historical AUC then BCE tie-break')
    temp=filename.with_suffix('.tmp')
    torch.save(ckpt,temp)
    temp.replace(filename)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['init','train'],required=True)
    p.add_argument('--arm',choices=['R-BCE','R-Debias'])
    p.add_argument('--root',type=Path,default=ROOT_DEFAULT)
    p.add_argument('--train-manifest',type=Path,default=Path('distillation/data_manifests/sd14_train20k_seed20261008.csv'))
    p.add_argument('--out-root',type=Path,default=Path('distillation/p0_matched_pair_20k_v1'))
    p.add_argument('--init-file',type=Path,default=Path('distillation/p0_matched_pair_20k_v1/shared_seed42_init.pt'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--max-steps',type=int,default=2625)
    p.add_argument('--val-every',type=int,default=175)
    p.add_argument('--rank',type=int,default=8)
    p.add_argument('--alpha',type=float,default=8.)
    p.add_argument('--augmentation',choices=['classic'],default='classic')
    p.add_argument('--dry-run',action='store_true',help='Single batch and gradient check, no optimizer step or output files')
    args=p.parse_args()
    if args.stage=='train' and not args.arm:p.error('--arm required for train')
    if args.stage=='init' and args.arm:p.error('omit --arm during init')
    if args.workers<0:p.error('workers must be nonnegative')
    return args


def main():
    args=parse_args()
    root=args.root.resolve()
    args.train_manifest=as_abs(root,args.train_manifest).resolve()
    args.out_root=as_abs(root,args.out_root).resolve()
    args.init_file=as_abs(root,args.init_file).resolve()
    args.run_name=f'{args.arm}_matched20k' if args.arm else 'shared_seed42_init'
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for exact P0 replication')
    if args.stage=='init' and args.init_file.exists() and not args.dry_run:
        raise RuntimeError('Shared init already exists; refusing overwrite')
    output=args.out_root/('R_BCE' if args.arm=='R-BCE' else 'R_Debias') if args.arm else None
    if output and output.exists() and not args.dry_run:
        raise RuntimeError(f'Experiment output exists, refusing overwrite: {output}')
    verify_old_files(root)
    verify_historical_p0_recipe(root,args)
    # Original deterministic initialization order.
    sys.path.insert(0,str(root/'distillation'))
    from train_early9_lora import seed_everything
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    old,v2,model,train_set,val_set=model_and_data(root,args)
    sig=signature(root,args,old,train_set,val_set)
    print('EXPERIMENT SIGNATURE:',json.dumps(sig,indent=2),flush=True)
    assert len(train_set)==20000 and len(val_set)==600
    # Keep both DataLoader construction and generator behavior exactly historical.
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,
        num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,
        generator=torch.Generator().manual_seed(args.seed))
    val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,
        num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    initial=val_metrics(model,val_loader)
    if abs(initial['auc']-.9788888889)>.0005:
        raise RuntimeError(f'Initial AUC does not match historical pre-P0: {initial["auc"]}')
    if args.stage=='init':
        shared=pack_init(model,sig)
        print('INIT VAL:',json.dumps(initial),flush=True)
        print('INIT WEIGHT SHA256:',shared['initial_weights_sha256'],flush=True)
        if args.dry_run:
            print('P0 MATCHED PAIR INIT DRY RUN: PASS (no files)',flush=True)
            return
        args.init_file.parent.mkdir(parents=True,exist_ok=True)
        torch.save(shared,args.init_file)
        print('SHARED INIT SAVED:',args.init_file,flush=True)
        print('SHARED INIT FILE SHA256:',sha256(args.init_file),flush=True)
        print('P0 MATCHED PAIR INITIALIZATION: PASS',flush=True)
        return
    if not args.init_file.is_file():raise FileNotFoundError('Run --stage init before train')
    init_sha=sha256(args.init_file)
    shared=torch.load(args.init_file,map_location='cpu',weights_only=False)
    load_shared(model,shared,sig)
    print('MATCHED INITIAL VALIDATION:',json.dumps(initial),flush=True)
    print('ARM:',args.arm,'| MATCHED RECIPE:',json.dumps({
        'max_steps':args.max_steps,'val_every':args.val_every,'seed':args.seed,
        'batch':args.batch_size,'lr_all':args.lr,'weight_decay':args.weight_decay,
        'augmentation':args.augmentation,'workers':args.workers,'debias_config':CFG if args.arm=='R-Debias' else None},sort_keys=True),flush=True)
    trainable=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable)!=307713:
        raise RuntimeError('Expected 307713 trained parameters from historical script')
    optimizer=torch.optim.AdamW(trainable,lr=args.lr,weight_decay=args.weight_decay)
    if args.dry_run:
        model.train();model.backbone.eval();model.head.train()
        images,labels=next(iter(train_loader))
        images=images.to('cuda:0');labels=labels.to('cuda:0')
        logits,scores,_=model(images)
        total,cls,edge=compute_loss(logits,scores,labels,args.arm)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        v2.check_lora_gradients(model)
        h=sum(float(p.grad.detach().abs().sum()) for p in model.head.parameters() if p.grad is not None)
        assert h>0 and torch.isfinite(total).item()
        print('DRYRUN LOSSES',{'total':float(total),'bce':float(cls),'edge':float(edge)},flush=True)
        print('DRYRUN EDGE MASS',fake_mass_stats(scores,labels),flush=True)
        print('P0 MATCHED PAIR DRY RUN: PASS (no step, no files)',flush=True)
        return
    output.mkdir(parents=True,exist_ok=False)
    best=dict(initial);best_step=0;best_check=0;history=[];validation_index=0
    checkpoint(model,old,args,args.arm,0,0,best,sig,init_sha,output/'best.pt')
    history.append(dict(step=0,epoch=0,validation_index=0,train_loss=None,train_cls_loss=None,
        train_edge_loss=None,head_training_at_val_start=False,**initial))
    global_step=0;epoch=0;interval_losses=[];interval_cls=[];interval_edge=[];interval_mass=[]
    head_train_steps=0;head_eval_steps=0
    while global_step < args.max_steps:
        epoch+=1
        model.train();model.backbone.eval();model.head.train()
        for images,labels in train_loader:
            was_training=bool(model.head.training)
            if was_training:head_train_steps+=1
            else:head_eval_steps+=1
            images=images.to('cuda:0',non_blocking=True)
            labels=labels.to('cuda:0',non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits,raw_scores,_=model(images)
            total,cls,edge=compute_loss(logits,raw_scores,labels,args.arm)
            if not bool(torch.isfinite(total)):
                raise RuntimeError(f'Nonfinite loss at step {global_step+1}')
            total.backward()
            if global_step==0:
                v2.check_lora_gradients(model)
            torch.nn.utils.clip_grad_norm_(trainable,max_norm=5.0)
            optimizer.step()
            interval_losses.append(float(total.item()))
            interval_cls.append(float(cls.item()))
            interval_edge.append(float(edge.item()))
            if args.arm=='R-Debias':
                interval_mass.append(fake_mass_stats(raw_scores,labels))
            global_step+=1
            if global_step%args.val_every==0 or global_step==args.max_steps:
                validation_index+=1
                metrics=val_metrics(model,val_loader)
                row=dict(step=global_step,epoch=epoch,validation_index=validation_index,
                    train_loss=float(np.mean(interval_losses)),train_cls_loss=float(np.mean(interval_cls)),
                    train_edge_loss=float(np.mean(interval_edge)),
                    head_training_at_val_start=was_training,
                    train_fake_edge_stats=merge_mass(interval_mass) if interval_mass else None,
                    **metrics)
                history.append(row)
                interval_losses.clear();interval_cls.clear();interval_edge.clear();interval_mass.clear()
                selected=improved(metrics,best)
                if selected:
                    best=dict(metrics);best_step=global_step;best_check=validation_index
                    checkpoint(model,old,args,args.arm,epoch,global_step,metrics,sig,init_sha,output/'best.pt')
                print(f'{args.arm} step={global_step}/{args.max_steps} epoch={epoch} '
                      f'AUC={metrics["auc"]:.6f} B1top1={metrics["fake_top1_b1"]:.4f} '
                      f'B1top10={metrics["fake_top10_b1"]:.4f} '
                      f'loss={row["train_loss"]:.6f} edge={row["train_edge_loss"]:.6f} '
                      f'BEST={selected}',flush=True)
                # Historical behavior: validation sets model.eval(), and head
                # stays in eval mode until the START of the following epoch.
            if global_step>=args.max_steps:break
    checkpoint(model,old,args,args.arm,epoch,global_step,metrics,sig,init_sha,output/'final_step.pt')
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,dict) else v for k,v in r.items()} for r in history]).to_csv(output/'history.csv',index=False)
    (output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    summary=dict(status='completed',arm=args.arm,shared_init_sha256=init_sha,
        shared_weight_sha256=shared['initial_weights_sha256'],signature=sig,
        best_step=best_step,best_validation_index=best_check,best_val_metrics=best,
        final_step=global_step,final_val_metrics=metrics,initial_val_metrics=initial,
        val_checks=validation_index,head_dropout_train_steps=head_train_steps,
        head_dropout_eval_steps=head_eval_steps,
        training_mode_note='Intentionally preserves original head eval-mode after validation',
        selection='historical AUC, BCE tie-break',
        note='No AIGIBench/GenImage metrics used for training or checkpoint selection')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('P0 MATCHED PAIR TRAINING: PASS',args.arm,'BEST STEP',best_step,
          'AUC',best['auc'],'B1',best['fake_top1_b1'],
          'DROPOUT_TRAIN_STEPS',head_train_steps,flush=True)

if __name__=='__main__':
    main()
