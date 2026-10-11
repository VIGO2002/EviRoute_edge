"""Paired 20k study: original EviRoute score dropout and fake-only edge mass.
Pure torch functions; no project, timm, filesystem, or GPU dependencies.
"""
import torch
import torch.nn.functional as F

CFG = dict(token_dropout=.15, edge1_dropout=.50, edge2_only_dropout=.25,
           edge_tau=1., lambda_edge1=.01, rho_edge1=.40,
           lambda_edge2=0., rho_edge2=.60)


def masks(grid=14, device='cpu'):
    y, x = torch.meshgrid(torch.arange(grid, device=device),
                          torch.arange(grid, device=device), indexing='ij')
    b1 = ((y == 0) | (y == grid-1) | (x == 0) | (x == grid-1)).reshape(-1)
    b2 = ((y <= 1) | (y >= grid-2) | (x <= 1) | (x >= grid-2)).reshape(-1)
    return b1, b2, b2 & ~b1


def masked_scores(scores, cfg=CFG):
    if scores.ndim != 2 or scores.shape[1] != 196:
        raise ValueError('Expected 196 patch scores')
    b, n = scores.shape
    mask = torch.zeros((b, n), dtype=torch.bool, device=scores.device)
    if cfg['token_dropout']:
        mask |= torch.rand((b, n), device=scores.device) < cfg['token_dropout']
    b1, _, b2_only = masks(device=scores.device)
    if cfg['edge1_dropout']:
        mask |= (torch.rand((b,n), device=scores.device) < cfg['edge1_dropout']) & b1[None,:]
    if cfg['edge2_only_dropout']:
        mask |= (torch.rand((b,n), device=scores.device) < cfg['edge2_only_dropout']) & b2_only[None,:]
    fully_masked = mask.all(dim=1)
    if bool(fully_masked.any()):
        mask[fully_masked, :] = False
    return scores.masked_fill(mask, -1e4)


def edge_loss(raw_scores, labels, cfg=CFG):
    fake = labels.reshape(-1) > .5
    if not bool(fake.any()) or (cfg['lambda_edge1'] <= 0 and cfg['lambda_edge2'] <= 0):
        return raw_scores.new_zeros(())
    p = torch.softmax(raw_scores[fake] / cfg['edge_tau'], dim=1)
    b1,b2,_ = masks(device=raw_scores.device)
    value = raw_scores.new_zeros(())
    if cfg['lambda_edge1'] > 0:
        mass = p[:,b1].sum(1)
        value = value + cfg['lambda_edge1'] * F.relu(mass-cfg['rho_edge1']).square().mean()
    if cfg['lambda_edge2'] > 0:
        mass = p[:,b2].sum(1)
        value = value + cfg['lambda_edge2'] * F.relu(mass-cfg['rho_edge2']).square().mean()
    return value


def compute_loss(logits, raw_scores, labels, arm, cfg=CFG):
    if arm == 'R-BCE':
        cls = F.binary_cross_entropy_with_logits(logits, labels)
        return cls, cls, logits.new_zeros(())
    if arm != 'R-Debias':
        raise ValueError('arm must be R-BCE or R-Debias')
    masked = masked_scores(raw_scores, cfg)
    masked_logits = masked.topk(k=10, dim=1).values.mean(1)
    cls = F.binary_cross_entropy_with_logits(masked_logits, labels)
    edge = edge_loss(raw_scores,labels,cfg)
    return cls+edge,cls,edge


@torch.no_grad()
def fake_mass_stats(raw_scores,labels,cfg=CFG):
    fake = labels.reshape(-1) > .5
    cnt = int(fake.sum().item())
    if cnt == 0:
        return dict(fake_count=0, active_count=0, mass_sum=0., hinge2_sum=0.)
    p=torch.softmax(raw_scores.detach()[fake].float()/cfg['edge_tau'],dim=1)
    b1,_,_=masks(device=raw_scores.device)
    masses=p[:,b1].sum(1)
    excess=F.relu(masses-cfg['rho_edge1'])
    return dict(fake_count=cnt, active_count=int((excess>0).sum().item()),
                mass_sum=float(masses.sum().item()),
                hinge2_sum=float(excess.square().sum().item()))


def merge_mass(records):
    summed = {k:sum(float(r[k]) for r in records) for k in ('fake_count','active_count','mass_sum','hinge2_sum')}
    n=summed['fake_count']
    return dict(fake_count=int(n), b1_mass_mean=None if n==0 else summed['mass_sum']/n,
                b1_active_fraction=None if n==0 else summed['active_count']/n,
                weighted_edge_loss_per_fake=None if n==0 else CFG['lambda_edge1']*summed['hinge2_sum']/n)


def improved(metrics,best):
    """Exact historical AUC -> BCE selection rule."""
    return metrics['auc'] > best['auc'] + 1e-6 or (
        abs(metrics['auc']-best['auc']) <= 1e-6 and metrics['bce'] < best['bce']-1e-6)
