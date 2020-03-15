
from .regularizer_l1 import L1Regularizer

class RegularizerManager(object):
    regularizers = {
        L1Regularizer.KEY: L1Regularizer
    }

    @classmethod
    def get_regularizer(cls, regularizer_config):
        regularizer = None
        if regularizer_config.l1_enabled:
            regularizer = RegularizerManager.regularizers.get(L1Regularizer.KEY)(regularizer_config)
        return regularizer
