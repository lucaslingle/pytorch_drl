import numpy as np
import gym

from drl.utils.types import Observation, EnvOutput
from drl.envs.wrappers.stateless import RewardSpec


class LockstepEnv(gym.core.Env):
    def __init__(self, cardinality: int = 100):
        super().__init__()
        self._observation_space_cardinality = cardinality
        self._observation_space = gym.spaces.Discrete(cardinality)
        self._action_space = gym.spaces.Discrete(cardinality + 1)
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

    def step(self, action: int) -> EnvOutput:
        assert action in self._action_space
        reward = float(self._state + 1) if action == self._state else 0.
        if action == self._state:
            new_state = self._state + 1
        else:
            new_state = self._state
        new_state %= self._observation_space_cardinality
        self._state = new_state

        o_tp1 = self._state
        r_t = {'extrinsic': reward, 'extrinsic_raw': reward}
        d_t = False
        i_t = {}
        return np.array(o_tp1), r_t, d_t, i_t
