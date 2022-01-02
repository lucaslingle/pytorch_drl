import abc

import torch as tc
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.algos.abstract import Algo
from drl.agents.preprocessing.common import EndToEndPreprocessing
from drl.agents.architectures import get_architecture
from drl.agents.integration import dynamic_mixin
from drl.utils.optim_util import get_optimizer


class PPO(Algo):
    def __init__(self, config):
        super().__init__(config)
        self._learning_system = self._get_learning_system()

    @abc.abstractmethod
    def _get_learning_system(self):
        preprocessing_spec = self.config.get('preprocessing')
        preprocessing = EndToEndPreprocessing(preprocessing_spec) # maybe do mixin for preprocessing too

        policy_config = self._config.get('policy_net')
        policy_config.get('architecture_cls_name')
        policy_architecture = get_architecture(
            cls_name=policy_config.get('architecture_cls_name'),
            cls_args=policy_config.get('architecture_cls_args'))
        dynamic_mixin(
            obj=policy_architecture,
            cls=get_head_cls(cls_name=policy_config.get('head_cls_name')),
            cls_args=policy_config.get('head_cls_args'))

        value_config = self._config.get('value_net')
        if value_config.get('use_separate_architecture'):
            value_architecture = get_architecture(
                cls_name=value_config.get('architecture_cls_name'),
                cls_args=value_config.get('architecture_cls_args'))
        else:
            value_architecture = policy_architecture
        dynamic_mixin(
            obj=value_architecture,
            cls=get_head_cls(cls_name=value_config.get('head_cls_name')),
            cls_args=value_config.get('head_cls_args'))

        # todo(lucaslingle):
        #   figure out a good way to mixin the preprocessing too.
        #   technically it comes first,
        #   but i kinda dont want the architecture to be a mixin.
        #   alternatively, maybe i start with an empty class and mix everything into it.

    @abc.abstractmethod
    def _train_loop(self):
        pass

    @abc.abstractmethod
    def _evaluation_loop(self):
        pass

    @abc.abstractmethod
    def _save_checkpoints(self):
        pass

    @abc.abstractmethod
    def _maybe_load_checkpoints(self):
        pass