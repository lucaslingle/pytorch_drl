import torch as tc
import numpy as np

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.algos.common import global_mean


class ReturnAcc(tc.nn.Module):
    # todo(lucaslingle):
    #      consider using this algorithm instead
    #      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, gamma, clip_low, clip_high):
        super().__init__()
        self._gamma = gamma
        self._clip_low = clip_low
        self._clip_high = clip_high
        self._current_ep_rewards = []
        self.register_buffer('_steps', tc.tensor(0))
        self.register_buffer('_mean', tc.tensor(0.))
        self.register_buffer('_var', tc.tensor(1.))

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, value):
        self.register_buffer('_steps', tc.tensor(value))

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        self.register_buffer('_mean', tc.tensor(value))

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, value):
        self.register_buffer('_var', tc.tensor(value))

    def update(self, r_t, d_t):
        self._current_ep_rewards.append(r_t)
        if d_t:
            returns = []
            while len(self._current_ep_rewards) > 0:
                r_t = self._current_ep_rewards.pop()
                if len(returns) == 0:
                    R_t = r_t
                else:
                    R_tp1 = returns[-1]
                    R_t = r_t + self._gamma * R_tp1
                returns.append(R_t)

            ep_steps = len(returns)
            steps = self.steps.item() + ep_steps

            returns = np.array(returns)

            mean = self.mean.item()
            mean *= ((steps-ep_steps) / steps)
            ep_ret_mean = np.mean(returns)
            mean += (ep_steps / steps) * ep_ret_mean

            var = self.var.item()
            var *= ((steps-ep_steps) / steps)
            ep_ret_var = np.mean(np.square(returns - np.expand_dims(mean, 0)))
            var += (ep_steps / steps) * ep_ret_var

            self.steps = steps
            self.mean = mean
            self.var = var

    def forward(self, r_t, shift=False, scale=True, eps=1e-4):
        mean, var = self.mean.unsqueeze(0), self.var.unsqueeze(0)
        if shift:
            r_t -= mean
        if scale:
            r_t *= tc.rsqrt(var + eps)
        if self._clip_low is not None:
            low = self._clip_low * tc.ones_like(r_t)
            r_t = tc.max(low, r_t)
        if self._clip_high is not None:
            high = self._clip_high * tc.ones_like(r_t)
            r_t = tc.min(r_t, high)
        return r_t


class NormalizeRewardWrapper(TrainableWrapper):
    def __init__(self, env, gamma, world_size, key=None):
        super().__init__(env)
        self._synced_normalizer = ReturnAcc(gamma, -10, 10)
        self._unsynced_normalizer = ReturnAcc(gamma, -10, 10)
        self._key = key
        self._world_size = world_size
        self._set_reward_spec()

    def _set_reward_spec(self):
        def spec_exists():
            if isinstance(self.env, Wrapper):
                return self.env.reward_spec is not None
            return False
        if not spec_exists():
            keys = ['extrinsic_raw', 'extrinsic']
            self.reward_spec = RewardSpec(keys)

    def _sync_normalizers_global(self):
        self._synced_normalizer.steps = global_mean(
            self._unsynced_normalizer.steps, self._world_size)
        self._synced_normalizer.mean = global_mean(
            self._unsynced_normalizer.mean, self._world_size)
        self._synced_normalizer.var = global_mean(
            self._unsynced_normalizer.var, self._world_size)

    def _sync_normalizers_local(self):
        self._unsynced_normalizer.steps = self._synced_normalizer.steps
        self._unsynced_normalizer.mean = self._synced_normalizer.mean
        self._unsynced_normalizer.var = self._synced_normalizer.var

    def get_checkpointables(self):
        checkpointables = dict()
        if isinstance(self.env, Wrapper):
            checkpointables.update(self.env.get_checkpointables())
        checkpointables.update({'reward_normalizer': self._synced_normalizer})
        return checkpointables

    def step(self, ac):
        if self._unsynced_normalizer.steps < self._synced_normalizer.steps:
            self._sync_normalizers_local()
        obs, rew, done, info = self.env.step(ac)

        if self._key:
            if self._key == 'extrinsic_raw':
                msg = "The key 'extrinsic_raw' must be preserved for logging."
                raise ValueError(msg)
            if not isinstance(rew, dict):
                msg = "Can't use non-keyed reward with keyed normalization wrapper."
                raise TypeError(msg)
        else:
            if isinstance(rew, dict):
                msg = "Can't use keyed reward w/ non-keyed normalization wrapper."
                raise TypeError(msg)

        reward = rew if not isinstance(rew, dict) else rew[self._key]
        reward = tc.tensor(reward).float()
        normalized = self._synced_normalizer(reward.unsqueeze(0)).item()
        self._unsynced_normalizer.update(reward, done)
        if isinstance(rew, dict):
            rew[self._key] = normalized
        else:
            rew = {'extrinsic_raw': reward, 'extrinsic': normalized}
        return obs, rew, done, info

    def learn(self, **kwargs):
        self._sync_normalizers_global()
        self._sync_normalizers_local()
