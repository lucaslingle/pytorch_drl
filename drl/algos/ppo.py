import warnings

import gym
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.algos.abstract import Algo
from drl.agents.preprocessing import EndToEndPreprocessing
from drl.agents.integration import (
    get_architecture_cls, get_head_cls, dynamic_mixin
)
from drl.utils.optim_util import get_optimizer
from drl.envs.wrappers.atari import AtariWrapper, DeepmindWrapper
from drl.envs.wrappers.common import ToTensorWrapper
# todo(lucaslingle):
#    bubble all wrappers up to the wrappers __init__ level,
#    so that the import has only three levels


class PPO(Algo):
    def __init__(self, rank, config):
        super().__init__(rank, config)
        self._learning_system = self._get_learning_system()
        print("Got learning system!!!")

    @staticmethod
    def _get_policy_net(policy_config):
        preprocessing_spec = policy_config.get('preprocessing_spec')
        preprocessing = EndToEndPreprocessing(preprocessing_spec)
        architecture_cls_name = policy_config.get('architecture_cls_name')
        architecture_cls = get_architecture_cls(architecture_cls_name)
        policy_net = architecture_cls(
            preprocessing, **policy_config.get('architecture_cls_args'))
        dynamic_mixin(
            obj=policy_net,
            cls=get_head_cls(policy_config.get('head_cls_name')),
            cls_args=policy_config.get('head_cls_args'))
        return policy_net

    @staticmethod
    def _get_value_net(value_config, policy_net):
        shared = value_config.get('use_shared_architecture')
        if shared:
            value_net = policy_net
        else:
            preprocessing_spec = value_config.get('preprocessing_spec')
            preprocessing = EndToEndPreprocessing(preprocessing_spec)
            architecture_cls_name = value_config.get('architecture_cls_name')
            architecture_cls = get_architecture_cls(architecture_cls_name)
            value_net = architecture_cls(
                preprocessing, **value_config.get('architecture_cls_args'))
        dynamic_mixin(
            obj=value_net,
            cls=get_head_cls(value_config.get('head_cls_name')),
            cls_args=value_config.get('head_cls_args'))
        return None if shared else value_net

    @staticmethod
    def _get_policy_optimizer(policy_config, policy_net):
        policy_optimizer = get_optimizer(
            model=policy_net,
            optimizer_cls_name=policy_config.get('optimizer_cls_name'),
            optimizer_cls_args=policy_config.get('optimizer_cls_args'))
        return policy_optimizer

    @staticmethod
    def _get_value_optimizer(value_config, value_net):
        shared = value_config.get('use_shared_architecture')
        if shared:
            return None
        value_optimizer = get_optimizer(
            model=value_net,
            optimizer_cls_name=value_config.get('optimizer_cls_name'),
            optimizer_cls_args=value_config.get('optimizer_cls_args'))
        return value_optimizer

    def _get_learning_system(self):
        policy_config = self._config.get('policy_net')
        value_config = self._config.get('value_net')

        policy_net = self._get_policy_net(policy_config)
        value_net = self._get_value_net(value_config, policy_net)

        policy_net = DDP(policy_net)
        if value_net:
            value_net = DDP(value_net)

        policy_optimizer = self._get_policy_optimizer(policy_config, policy_net)
        value_optimizer = self._get_value_optimizer(value_config, value_net)

        # todo(lucaslingle):
        #  Support for wrapping via config,
        #    similar to pytorch_ddp_resnet for transforms.
        warnings.warn("PPO currently only supports atari games!", UserWarning)
        env = gym.make(self._config.get('env_id'))
        env = ToTensorWrapper(DeepmindWrapper(AtariWrapper(env), frame_stack=False))

        checkpointables = {
            'policy_net': policy_net,
            'value_net': value_net,
            'policy_optimizer': policy_optimizer,
            'value_optimizer': value_optimizer
        }
        checkpointables_ = {k: v for k,v in checkpointables.items()}
        checkpointables_.update(env.get_checkpointables())
        global_step = self._maybe_load_checkpoints(checkpointables_, step=None)
        return {'global_step': global_step, 'env': env, **checkpointables}

    def _compute_losses(self, mb):
        # todo(lucaslingle): when defining compute losses,
        #  be sure to check config.use_separate_architecture
        raise NotImplementedError

    def training_loop(self):
        raise NotImplementedError

    def evaluation_loop(self):
        raise NotImplementedError
