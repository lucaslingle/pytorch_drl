import warnings

import gym
from torch.nn.parallel import DistributedDataParallel as DDP

from drl.algos.abstract import Algo
from drl.agents.preprocessing import EndToEndPreprocessing
from drl.agents.integration import get_architecture, get_predictors, Agent
from drl.utils.optim_util import get_optimizer
from drl.envs.wrappers.integration import get_wrappers
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
    def _get_opt(opt_config, agent):
        optimizer = get_optimizer(model=agent, **opt_config)
        return optimizer

    @staticmethod
    def _get_env(env_config):
        env = gym.make(env_config.get('id'))
        env = get_wrappers(env, env_config.get('wrappers'))
        return env

    def _get_learning_system(self):
        pol_config = self._config.get('policy_net')
        val_config = self._config.get('value_net')
        shared = val_config.get('use_shared_architecture')

        pol_net = self._get_net(pol_config)
        pol_opt = self._get_opt(pol_config.get('optimizer'), pol_net)

        if not shared:
            val_net = self._get_net(val_config)
            val_opt = self._get_opt(val_config.get('optimizer'), val_net)
        else:
            val_net = None
            val_opt = None

        env_config = self._config.get('env')
        env = self._get_env(env_config)

        checkpointables = {
            'policy_net': pol_net,
            'value_net': val_net,
            'policy_optimizer': pol_opt,
            'value_optimizer': val_opt
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
