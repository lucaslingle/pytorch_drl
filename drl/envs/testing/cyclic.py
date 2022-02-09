import numpy as np
import gym

from drl.utils.typing import Observation, Action, EnvOutput
from drl.envs.wrappers.stateless import RewardSpec


class CyclicEnv(gym.core.Env):
    def __init__(self):
        super().__init__()
        self._observation_space_cardinality = 100
        self._observation_space = gym.spaces.Discrete(
            self._observation_space_cardinality)
        self._action_space = gym.spaces.Discrete(1)
        self._initial_state = 0
        self._state = self._initial_state

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_spec(self) -> RewardSpec:
        return RewardSpec(keys=['extrinsic', 'extrinsic_raw'])

    def reset(self) -> Observation:
        self._state = self._initial_state
        return np.array(self._state)

    def step(self, action: Action) -> EnvOutput:
        new_state = self._state + 1
        new_state %= self._observation_space_cardinality
        self._state = new_state

        o_tp1 = self._state
        r_t = {'extrinsic': 1.0, 'extrinsic_raw': 1.0}
        d_t = False
        i_t = {}
        return np.array(o_tp1), r_t, d_t, i_t
