import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.losses import contrastive_loss, norm_mse_loss, distributed_sinkhorn, shoot_infs

def get_K(all_perframe):
    prev = all_perframe[:,:,:-1]
    prev = F.normalize(prev, p=2, dim=1)
    post = all_perframe[:,:,1:]
    post = F.normalize(post, p=2, dim=1)
    prev_pinv = torch.linalg.pinv(prev, rtol=1e-4)
    all_K = torch.matmul(post, prev_pinv)
    return all_K

def split_feature(K, Z, threshold=0.1):
    B = K.shape[0]

    eigenvalues, eigenvectors = torch.linalg.eig(K)  # eigenvectors: (B, M, M)

    eigenvectors = torch.linalg.qr(eigenvectors).Q  # shape: (B, M, M)

    distance_to_one = torch.abs(eigenvalues - 1.0)
    static_mask_indices = []
    dynamic_mask_indices = []

    for b in range(B):
        static_mask_indices.append(torch.where(distance_to_one[b] <= threshold)[0])
        dynamic_mask_indices.append(torch.where(distance_to_one[b] > threshold)[0])

    # Z shape: (B, L, M) → Gx: (B, L, M)
    Gx = Z.permute(0, 2, 1)

    static = stratify(eigenvalues, eigenvectors, Gx, static_mask_indices).mean(dim=1)
    dynamic = stratify(eigenvalues, eigenvectors, Gx, dynamic_mask_indices).mean(dim=1)

    return static, dynamic

def stratify(eigenvalues, eigenvectors, Gx, mask_indices):
    B, L, M = Gx.shape  # Gx: (B, L, M)
    device = Gx.device

    eigenvalues = eigenvalues.to(device)
    eigenvectors = eigenvectors.to(device)

    Lambda = torch.zeros(B, M, M, dtype=eigenvalues.dtype, device=device)
    for b in range(B):
        Lambda[b] = torch.diag(eigenvalues[b])

    masked_Lambda = torch.zeros_like(Lambda)
    for b in range(B):
        mask = torch.zeros(M, dtype=eigenvalues.dtype, device=device)
        mask[mask_indices[b]] = 1.0
        masked_Lambda[b] = torch.diag(mask) @ Lambda[b]

    try:
        Phi_inv = torch.inverse(eigenvectors)
    except RuntimeError:
        Phi_inv = torch.pinverse(eigenvectors)

    Gx_reshaped = Gx.reshape(B * L, M)

    Phi_expanded = eigenvectors.unsqueeze(1).expand(B, L, M, M).reshape(B * L, M, M)
    masked_Lambda_expanded = masked_Lambda.unsqueeze(1).expand(B, L, M, M).reshape(B * L, M, M)
    Phi_inv_expanded = Phi_inv.unsqueeze(1).expand(B, L, M, M).reshape(B * L, M, M)

    reconstructed_Gx = (Phi_expanded @ masked_Lambda_expanded @ Phi_inv_expanded).real @ Gx_reshaped.unsqueeze(-1)
    reconstructed_Gx = reconstructed_Gx.squeeze(-1).reshape(B, L, M)

    return reconstructed_Gx

def train_simclr_koop(model, clips, args):
    all_perframe = []
    for clip in clips:
        all_perframe.append(model.get_perframe_representation(clip))
    static_loss = 0
    dynamic_loss = 0
    perframe_first = all_perframe[0]
    perframe_second = all_perframe[1]
    K_first = get_K(perframe_first)
    K_second = get_K(perframe_second)
    static_first, dynamic_first = split_feature(K_first, perframe_first)
    static_second, dynamic_second = split_feature(K_second, perframe_second)
    static_loss += contrastive_loss(static_first, static_second,args.static_tau)
    dynamic_loss += contrastive_loss(dynamic_first, dynamic_second,args.dynamic_tau)
    total_loss = static_loss + dynamic_loss
    return total_loss

def train_byol_koop(model, clips, args):
    x_1, x_2 = clips
    z1 = model.get_perframe_representation(x_1)
    zt_1 = model.get_perframe_representation_target(x_1)
    z2 = model.get_perframe_representation(x_2)
    zt_2 = model.get_perframe_representation_target(x_2) 
    
    z1_K = get_K(z1)
    z2_K = get_K(z2)
    zt1_K = get_K(zt_1)
    zt2_K = get_K(zt_2)
    static1, dynamic1 = split_feature(z1_K, z1)
    static2, dynamic2 = split_feature(z2_K, z2)
    static_zt1, dynamic_zt1 = split_feature(zt1_K, zt_1)
    static_zt2, dynamic_zt2 = split_feature(zt2_K, zt_2)
    static_loss = 0
    dynamic_loss = 0
    static_loss += norm_mse_loss(static1, static_zt2,args.static_tau)
    static_loss += norm_mse_loss(static2, static_zt1,args.static_tau)
    dynamic_loss += norm_mse_loss(dynamic1, dynamic_zt2,args.dynamic_tau)
    dynamic_loss += norm_mse_loss(dynamic2, dynamic_zt1,args.dynamic_tau)
    total_loss = static_loss + dynamic_loss
    return total_loss   
    
def swav_loss(p1, p2, args):
    q1 = distributed_sinkhorn(torch.exp(p1 / args.epsilon).t(), args.n_iters)
    q2 = distributed_sinkhorn(torch.exp(p2 / args.epsilon).t(), args.n_iters)
    
    p1 = F.softmax(p1 / args.temperature, dim=1)
    p2 = F.softmax(p2 / args.temperature, dim=1)

    loss1 = -torch.mean(torch.sum(q1 * torch.log(p2), dim=1))
    loss2 = -torch.mean(torch.sum(q2 * torch.log(p1), dim=1))
    loss = loss1+loss2
    return loss

def train_swav_koop(model, clips, args):
    x_1, x_2 = clips
    z1 = model.get_perframe_representation(x_1)
    z2 = model.get_perframe_representation(x_2)
    z1_K = get_K(z1)
    z2_K = get_K(z2)
    static1, dynamic1 = split_feature(z1_K, z1)
    static2, dynamic2 = split_feature(z2_K, z2)
    static_loss = 0
    dynamic_loss = 0
    static_loss += swav_loss(static1, static2, args)
    dynamic_loss += swav_loss(dynamic1, dynamic2, args)
    total_loss = static_loss + dynamic_loss
    return total_loss

def moco_loss(z1, z2, queue, T=0.2):
    l_pos = torch.einsum('nc,nc->n', [z1, z2]).unsqueeze(-1)
    l_neg = torch.einsum('nc,kc->nk', [z1, queue.clone().detach()])
    logits = torch.cat([l_pos, l_neg], dim=1).div(T)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    loss = F.cross_entropy(logits, labels)
    return loss

def train_moco_koop(model, clips, args):
    x_1, x_2 = clips
    z1 = model.get_perframe_representation(x_1)
    zt2 = model.get_perframe_representation_target(x_2)
    z1_K = get_K(z1)
    zt2_K = get_K(zt2)
    static1, dynamic1 = split_feature(z1_K, z1)
    static2, dynamic2 = split_feature(zt2_K, zt2)
    static_loss = 0
    dynamic_loss = 0
    static_loss += moco_loss(static1, static2, model.queue)
    dynamic_loss += moco_loss(dynamic1, dynamic2, model.queue)
    total_loss = static_loss + dynamic_loss
    model.update_target(args.momentum)
    keys = z1
    # not excess the total length
    if model.queue.ptr+keys.shape[0] < args.K:
        model.queue[model.queue.ptr:model.queue.ptr+keys.shape[0]] = keys
    # circle queue
    else:
        model.queue[model.queue.ptr:args.K] = keys[:args.K-model.queue.ptr]
        model.queue[:keys.shape[0] - args.K + model.queue.ptr] = keys[args.K-model.queue.ptr:]
    model.queue.ptr = (model.queue.ptr+keys.shape[0]) % args.K
    
    return total_loss
    

def train_inner(model, clips, args):
    total_sample = torch.cat(clips, dim=0)
    all_perframe = model.get_perframe_representation(total_sample) # NxDxT
    prev = all_perframe[:,:,:-1]
    post = all_perframe[:,:,1:]
    prev_pinv = torch.linalg.pinv(prev, rtol=args.pinv_rtol)
    all_K = torch.matmul(post, prev_pinv)
    prev_detach = prev.detach()
    post_detach = post
    post_hat = torch.matmul(all_K, prev_detach)
    predict_loss = torch.norm(post_hat-post_detach, p='fro')
    return predict_loss