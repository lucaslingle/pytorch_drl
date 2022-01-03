import warnings

import gym
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.algos.abstract import Algo
from drl.agents.preprocessing import EndToEndPreprocessing
from drl.agents.integration import (
    get_architecture, get_predictors, Agent
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
    def _get_net(net_config):
        preprocessing = EndToEndPreprocessing(net_config.get('preprocessing'))
        architecture = get_architecture(**net_config.get('architecture'))
        predictors = get_predictors(**net_config.get('predictors'))
        return DDP(Agent(preprocessing, architecture, predictors))

    @staticmethod
    def _get_optim(config, agent):
        policy_optimizer = get_optimizer(
            model=agent,
            optimizer_cls_name=config.get('optimizer_cls_name'),
            optimizer_cls_args=config.get('optimizer_cls_args'))
        return policy_optimizer

    def _get_learning_system(self):
        pol_config = self._config.get('policy_net')
        val_config = self._config.get('value_net')
        shared = val_config.get('use_shared_architecture')

        pol_net = self._get_net(pol_config)
        val_net = self._get_net(val_config) if not shared else None

        pol_optim = self._get_optim(pol_config, pol_net)
        val_optim = self._get_optim(val_config, val_net) if not shared else None

        # todo(lucaslingle):
        #  Support for wrapping via config,
        #    similar to pytorch_ddp_resnet for transforms.
        warnings.warn("PPO currently only supports atari games!", UserWarning)
        env = gym.make(self._config.get('env_id'))
        env = ToTensorWrapper(DeepmindWrapper(AtariWrapper(env), frame_stack=False))

        checkpointables = {
            'policy_net': pol_net,
            'value_net': val_net,
            'policy_optimizer': pol_optim,
            'value_optimizer': val_optim
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
