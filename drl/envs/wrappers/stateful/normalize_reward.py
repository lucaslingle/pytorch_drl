import torch as tc

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.algos.common import global_mean


class ReturnAcc(tc.nn.Module):
    def __init__(self, gamma, clip_low, clip_high):
        super().__init__()
        self._gamma = gamma
        self._current_ep_rewards = []
        self._normalizer = Normalizer([], clip_low, clip_high)

    @property
    def steps(self):
        return self._normalizer.steps

    @steps.setter
    def steps(self, value):
        self._normalizer.steps = value

    @property
    def moment1(self):
        return self._normalizer.moment1

    @moment1.setter
    def moment1(self, tensor):
        self._normalizer.moment1 = tensor

    @property
    def moment2(self):
        return self._normalizer.moment2

    @moment2.setter
    def moment2(self, tensor):
        self._normalizer.moment2 = tensor

    def update(self, r_t, d_t):
        self._current_ep_rewards.append(r_t)
        if d_t:
            returns = list()
            while len(self._current_ep_rewards) > 0:
                r_t = self._current_ep_rewards.pop()
                if len(returns) == 0:
                    R_t = r_t
                else:
                    R_tp1 = returns[-1]
                    R_t = r_t + self._gamma * R_tp1
                returns.append(R_t)

            ep_steps = len(returns)
            steps = self._normalizer.steps + ep_steps

            returns = tc.tensor(returns)

            moment1 = self._normalizer.moment1
            moment1 *= ((steps-ep_steps) / steps)
            ep_ret_mean = tc.mean(returns)
            moment1 += (ep_steps / steps) * ep_ret_mean

            moment2 = self._normalizer.moment2
            moment2 *= ((steps-ep_steps) / steps)
            ep_ret_var = tc.mean(tc.square(returns))
            moment2 += (ep_steps / steps) * ep_ret_var

            self._normalizer.steps = steps
            self._normalizer.moment1 = moment1
            self._normalizer.moment2 = moment2

    def forward(self, r_t, shift=False, scale=True):
        return self._normalizer(r_t, shift=shift, scale=scale)


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
        self._synced_normalizer.moment1 = global_mean(
            self._unsynced_normalizer.moment1, self._world_size)
        self._synced_normalizer.moment2 = global_mean(
            self._unsynced_normalizer.moment2, self._world_size)

    def _sync_normalizers_local(self):
        self._unsynced_normalizer.steps = self._synced_normalizer.steps
        self._unsynced_normalizer.moment1 = self._synced_normalizer.moment1
        self._unsynced_normalizer.moment2 = self._synced_normalizer.moment2

    @property
    def checkpointables(self):
        checkpoint_dict = self.env.checkpointables
        checkpoint_dict.update({'reward_normalizer': self._synced_normalizer})
        return checkpoint_dict

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
