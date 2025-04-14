import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()


def contrastive_loss(x0, x1, tau=0.1):
    bsize = x0.shape[0]
    x0 = F.normalize(x0, p=2, dim=1)
    x1 = F.normalize(x1, p=2, dim=1)
    x0_large = x0
    x1_large = x1
    target = torch.arange(0, bsize).cuda()
    eye_mask = F.one_hot(target, num_classes=bsize).cuda() * 1e9

    logits00 = x0 @ x0_large.t() / tau - eye_mask
    logits11 = x1 @ x1_large.t() / tau - eye_mask
    logits01 = x0 @ x1_large.t() / tau
    logits10 = x1 @ x0_large.t() / tau
    return (
        F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
        + F.cross_entropy(torch.cat([logits10, logits11], dim=1), target)
    ) / 2
    
def moco_loss(z1, z2, queue, T=0.2):
    l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
    l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1).div(T)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss

def swav_loss(p1, p2, epsilon=0.05, n_iters=3, temperature=0.1):
    q1 = distributed_sinkhorn(torch.exp(p1 / epsilon).t(), n_iters)
    q2 = distributed_sinkhorn(torch.exp(p2 / epsilon).t(), n_iters)
    
    p1 = F.softmax(p1 / temperature, dim=1)
    p2 = F.softmax(p2 / temperature, dim=1)

    loss1 = -torch.mean(torch.sum(q1 * torch.log(p2), dim=1))
    loss2 = -torch.mean(torch.sum(q2 * torch.log(p1), dim=1))
    loss = loss1+loss2
    return loss

def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor

def distributed_sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        # c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (args.world_size * Q.shape[1])
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()