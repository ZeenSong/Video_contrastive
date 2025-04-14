from itertools import chain
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, S3D
from pytorchvideo.models.hub import slow_r50
from models.models_vit import vit_base_patch16

class R2Plus1D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = r2plus1d_18()
        self._remove_temporal_downsampling()
        
    def _remove_temporal_downsampling(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv3d):
                k = m.kernel_size
                s = m.stride
                p = m.padding
                if s[0] > 1:
                    m.stride = (1, s[1], s[2])
                if k[0] > 1 and p[0] > 0:
                    m.padding = (1, p[1], p[2])
            if isinstance(m, nn.MaxPool3d):
                s = m.stride
                p = m.padding
                k = m.kernel_size
                if isinstance(k, tuple) and k[0] > 1:
                    m.kernel_size = (1, k[1], k[2])
                if s[0] > 1:
                    m.stride = (1, s[1], s[2])
                if p[0] > 0:
                    m.padding = (0, p[1], p[2])
    
    def forward(self, x):
        x = self.backbone.stem(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        return x

def get_MLP(in_dim, out_dim, hidden_dim):
    mlp = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim, bias=False)
    )
    return mlp

def get_backbone(args):
    if args.backbone == 'slow_r50':
        backbone = slow_r50(head=None)
    elif args.backbone == 'vit':
        backbone = vit_base_patch16(
            num_frames=args.num_frames, t_patch_size=args.t_patch_size, img_size=args.img_size,
        )
    else:
        backbone = R2Plus1D()
    return backbone

class SimCLR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.args = args
        self.spatio_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.spatio_temporal_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.head = get_MLP(args.repre_dim, args.out_dim, args.hidden_dim)

    def forward(self, batch):
        if isinstance(batch, list):
            all_output = []
            for x in batch:
                x = x.cuda()
                r = self.backbone(x)
                if self.args.backbone != 'vit':
                    r = self.spatio_pool(r).squeeze()
                else:
                    r = r.mean(dim=1)
                z = self.head(r)
                all_output.append(z)
            return all_output
        if self.args.backbone != 'vit':
            output = self.spatio_temporal_pool(self.backbone(batch)).squeeze()
        else:
            output = self.backbone(batch).mean(dim=1)
        return output

    def get_representation(self, batch):
        return self.spatio_temporal_pool(self.backbone(batch)).squeeze()
    
    def get_perframe_representation(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.backbone(batch)).squeeze().permute(0,2,1)
        else:
            r = self.backbone(batch).mean(dim=1).permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.head(r)
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z
  

class BYOL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = get_backbone(args)
        self.target = get_backbone(args)
        self.head = get_MLP(args.repre_dim, args.out_dim, args.hidden_dim)
        self.predictor = get_MLP(args.out_dim, args.out_dim, args.hidden_dim)
        self.head_target = get_MLP(args.repre_dim, args.out_dim, args.hidden_dim)
        self.spatio_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.spatio_temporal_pool = nn.AdaptiveAvgPool3d(output_size=1)
        for param in chain(self.target.parameters(), self.head_target.parameters()):
            param.requires_grad = False
        self.byol_tau = args.byol_tau
        self.update_target(0)

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.target.parameters(), self.backbone.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_target.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)

    def forward(self, batch):
        if isinstance(batch, list):
            all_output = []
            all_target_output = []
            for x in batch:
                x = x.cuda()
                if self.args.backbone != 'vit':
                    z = self.predictor(self.head(self.spatio_temporal_pool(self.backbone(x)).squeeze()))
                else:
                    z = self.predictor(self.head(self.backbone(x).mean(dim=1)))
                z = F.normalize(z, dim=-1, p=2)
                all_output.append(z)
                with torch.no_grad():
                    if self.args.backbone != 'vit':
                        zt = self.head_target(self.spatio_temporal_pool(self.target(x)).squeeze())
                    else:
                        zt = self.head_target(self.target(x).mean(dim=1))
                    zt = F.normalize(zt, dim=-1, p=2)
                    all_target_output.append(zt)
            return all_output, all_target_output
        if self.args.backbone != 'vit':
            output = self.spatio_temporal_pool(self.backbone(batch)).squeeze()
        else:
            output = self.backbone(batch).mean(dim=1)
        return output

    def get_representation(self, batch):
        return self.spatio_temporal_pool(self.backbone(batch)).squeeze()
    
    def get_perframe_representation(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.backbone(batch)).squeeze().permute(0,2,1)
        else:
            r = self.backbone(batch).mean(dim=1).permute(0,2,1)
        # r = self.spatio_pool(self.backbone(batch)).squeeze().permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.predictor(self.head(r))
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z
    
    def get_perframe_representation_target(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.target(batch)).squeeze().permute(0,2,1)
        else:
            r = self.target(batch).mean(dim=1).permute(0,2,1)
        # r = self.spatio_pool(self.target(batch)).squeeze().permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.head_target(r)
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z

class Swav(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.args = args
        self.spatio_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.spatio_temporal_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.head = get_MLP(args.repre_dim, args.out_dim, args.hidden_dim)
        self.spatio_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.spatio_temporal_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.prototypes = nn.Linear(args.out_dim, 100, bias=False)
    
    def forward(self, batch):
        if isinstance(batch, list):
            all_output = []
            for x in batch:
                x = x.cuda()
                r = self.backbone(x)
                if self.args.backbone != 'vit':
                    r = self.spatio_temporal_pool(r).squeeze()
                else:
                    r = r.mean(dim=1)
                z = self.head(r)
                z = F.normalize(z, dim=-1, p=2)
                z = self.prototypes(z)
                all_output.append(z)
            return all_output
        if self.args.backbone != 'vit':
            output = self.spatio_temporal_pool(self.backbone(batch)).squeeze()
        else:
            output = self.backbone(batch).mean(dim=1)
        return output
    
    def get_representation(self, batch):
        return self.spatio_temporal_pool(self.backbone(batch)).squeeze()
    
    def get_perframe_representation(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.backbone(batch)).squeeze().permute(0,2,1)
        else:
            r = self.backbone(batch).mean(dim=1).permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.prototypes(self.head(r))
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z    
    

class MoCo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.target = get_backbone(args)
        self.queue = F.normalize(torch.randn(args.K, args.out_dim).cuda()).detach()
        self.queue.requires_grad = False
        self.queue.ptr = 0
        self.head = get_MLP(args.rep_dim, args.out_dim, args.hidden_dim)
        self.head_target = get_MLP(args.rep_dim, args.out_dim, args.hidden_dim)
        for param in chain(self.target.parameters(), self.head_target.parameters()):
            param.requires_grad = False
        self.spatio_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.temporal_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.spatio_temporal_pool = nn.AdaptiveAvgPool3d(output_size=1)
        
    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.target.parameters(), self.back_bone.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_target.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        
    def forward(self, batch):
        if isinstance(batch, list):
            all_output = []
            all_target_output = []
            for x in batch:
                x = x.cuda()
                if self.args.backbone != 'vit':
                    z = self.head(self.spatio_temporal_pool(self.backbone(x)).squeeze())
                else:
                    z = self.head(self.backbone(x).mean(dim=1))
                z = F.normalize(z, dim=-1, p=2)
                all_output.append(z)
                with torch.no_grad():
                    if self.args.backbone != 'vit':
                        zt = self.head_target(self.spatio_temporal_pool(self.target(x)).squeeze())
                    else:
                        zt = self.head_target(self.target(x).mean(dim=1))
                    zt = F.normalize(zt, dim=-1, p=2)
                    all_target_output.append(zt)
            return all_output, all_target_output
        if self.args.backbone != 'vit':
            output = self.spatio_temporal_pool(self.backbone(batch)).squeeze()
        else:
            output = self.backbone(batch).mean(dim=1)
        return output
    
    def get_representation(self, batch):
        return self.spatio_temporal_pool(self.backbone(batch)).squeeze()
    
    def get_perframe_representation(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.backbone(batch)).squeeze().permute(0,2,1)
        else:
            r = self.backbone(batch).mean(dim=1).permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.head(r)
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z    
    
    def get_perframe_representation_target(self, batch):
        # NxCxTxHxW -> NxTxD
        if self.args.backbone != 'vit':
            r = self.spatio_pool(self.target(batch)).squeeze().permute(0,2,1)
        else:
            r = self.target(batch).mean(dim=1).permute(0,2,1)
        # r = self.spatio_pool(self.target(batch)).squeeze().permute(0,2,1)
        N, T, D = r.shape
        r = r.reshape(N*T, D)
        z = self.head_target(r)
        D = z.shape[1]
        z = z.reshape(N,T,D).permute(0,2,1)
        return z
    
    