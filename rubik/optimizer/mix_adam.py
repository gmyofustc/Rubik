import torch.optim import optimizer
from rubik.optimizer.bmuf_optimizer import BMUFOptimizer

class Adam(BMUFOptimizer):
    def __init__(self, params, hparams):
        defaults = dict(lr=hparams.lr, betas=(hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay, bm_lr=hparams.bmuf_lr,
        bm_mom=hparams.bmuf_mom)
        super(MixAdam, self).__init__(params, defaults)
        self.hparams = hparams

    def step(self, closure=None):
        #USE Adam Step
