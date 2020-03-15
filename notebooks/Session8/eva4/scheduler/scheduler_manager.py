from .scheduler_steplr import StepLRScheduler
from .scheduler_multisteplr import MultiStepLRScheduler

class SchedulerManager(object):
    schedulers = {
        StepLRScheduler.KEY: StepLRScheduler,
        MultiStepLRScheduler.KEY: MultiStepLRScheduler,
    }

    @classmethod
    def get_scheduler(cls, optimizer, scheduler_config):
        scheduler = SchedulerManager.schedulers.get(scheduler_config.type)(optimizer, scheduler_config)
        return scheduler

