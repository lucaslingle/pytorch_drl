import torch as tc

from drl.envs.wrappers.stateless.abstract import Wrapper, RewardSpec
from drl.envs.wrappers.stateful.abstract import TrainableWrapper
from drl.envs.wrappers.stateful.normalize import Normalizer
from drl.algos.common import global_mean


class NormalizeRewardWrapper(TrainableWrapper):
    # todo(lucaslingle):
    #  (1) baseline's VecNormalize uses running discounted cumulative reward
    #        as part of state.
    #  (2) Need to add that, and rewrite normalization mechanism
    #        to update based on it.
    #  (3) For fault tolerance, make it a separate class with a torch buffer,
    #        and add it to the get_checkpointables return dict.
    def __init__(self, env, key=None):
        super().__init__(env)
        self._synced_normalizer = Normalizer((1,), -5, 5)
        self._unsynced_normalizer = Normalizer((1,), -5, 5)
        self._key = key
        self._set_reward_spec()

    def _set_reward_spec(self):
        def spec_exists():
            if isinstance(self._env, Wrapper):
                return self._env.reward_spec is not None
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
        if self._unsynced_normalizer.step < self._synced_normalizer.step:
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

        # todo(lucaslingle): add support here and in ClipRewardWrapper
        #  for 'all' key or perhaps change None to mean 'all'.
        reward = rew if not isinstance(rew, dict) else rew[self._key]
        reward = tc.tensor([reward]).float()
        normalized = self._synced_normalizer(reward.unsqueeze(0), shift=False).item()
        _ = self._unsynced_normalizer.update(reward)
        if isinstance(rew, dict):
            rew[self._key] = normalized
        else:
            rew = {'extrinsic_raw': reward, 'extrinsic': normalized}
        return obs, rew, done, info

    def learn(self, **kwargs):
        self._sync_normalizers_global()
        self._sync_normalizers_local()
