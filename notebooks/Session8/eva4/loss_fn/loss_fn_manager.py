from .loss_fn_cross_entropy_loss import CrossEntropyLossFn
from .loss_fn_nll_loss import NLLLossFn

class LossFnManager(object):
    losses = {
        CrossEntropyLossFn.KEY: CrossEntropyLossFn,
        NLLLossFn.KEY: NLLLossFn,
    }

    @classmethod
    def get_loss_fn(cls, loss_fn_config):
        loss = LossFnManager.losses.get(loss_fn_config.type)(loss_fn_config)
        return loss

