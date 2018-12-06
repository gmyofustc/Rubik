import torch.optim import optimizer

class BMUFOptimizer(Opitmizer):
    def __init__(self, params, hparams):
        defaults = dict(lr=hparams.lr, betas=(hparams.adam_beta1, hparams.adam_beta2),
        eps=hparams.adam_eps, weight_decay=hparams.weight_decay, bm_lr=hparams.bmuf_lr,
        bm_mom=hparams.bmuf_mom)
        super(MixAdam, self).__init__(params, defaults)
        self.hparams = hparams

    def step(self, closure=None):
        #USE Adam Step

    def bmuf_step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                param_buffer, delta_buffer = state['param_buffer'], state['delta_buffer']
                Gt = p.data-param_buffer
                delta_buffer.mul_(group['bm_mom']).add_(Gt.mul_(group['bm_lr']))
                param_buffer.add(delta_buffer)
                p.data.copy_(param_buffer)