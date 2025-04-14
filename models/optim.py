import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

class LARS(Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class MyScheduler(object):
    def __init__(self, warm_up_epoch: int, step_epochs: list, max_epochs: int, optimizer: Optimizer, step_scale: float, init_lr: float):
        self.warm_up_epoch = warm_up_epoch
        self.step_epochs = step_epochs
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.step_scale = step_scale
        self.cos_scheduler = CosineAnnealingLR(optimizer, self.max_epochs - self.warm_up_epoch, eta_min=1e-5)
        self.step_scheduler = MultiStepLR(optimizer, milestones=self.step_epochs, gamma=self.step_scale)
        self.init_lr = init_lr
    
    def step(self, cur_epoch:int):
        # warm up
        if cur_epoch < self.warm_up_epoch:
            lr_scale = (cur_epoch + 1) / self.warm_up_epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr * lr_scale
        else:
            self.cos_scheduler.step()
        self.step_scheduler.step()
