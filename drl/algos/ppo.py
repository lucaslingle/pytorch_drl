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
        env_config = self._config.get('env')
        env = self._get_env(env_config)

        policy_config = self._config.get('policy_net')
        policy_net = self._get_net(policy_config)
        policy_optimizer_config = policy_config.get('optimizer')
        policy_optimizer = self._get_opt(policy_optimizer_config, policy_net)

        value_config = self._config.get('value_net')
        value_net, value_optimizer = None, None
        if not value_config.get('use_shared_architecture'):
            value_net = self._get_net(value_config)
            value_optimizer_config = value_config.get('optimizer')
            value_optimizer = self._get_opt(value_optimizer_config, value_net)

        checkpointables = {
            'policy_net': policy_net,
            'policy_optimizer': policy_optimizer,
            'value_net': value_net,
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
