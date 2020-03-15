from .optimizer_sgd import SGDOptimizer

class OptimizerManager(object):
    optimizers = {
        SGDOptimizer.KEY: SGDOptimizer
    }

    @classmethod
    def get_optimizer(cls, model, optimizer_config):
        optimizer = OptimizerManager.optimizers.get(optimizer_config.type)(model, optimizer_config)
        return optimizer

