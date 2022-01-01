import torch as tc

from drl.envs.wrappers.common.abstract import ObservationWrapper, RewardWrapper
from drl.envs.wrappers.stateful.abstract import StatefulWrapper


class Normalizer(tc.nn.Module):
    def __init__(self, data_shape, clip_low=None, clip_high=None):
        super().__init__()
        self._data_shape = data_shape
        self._clip_low = clip_low
        self._clip_high = clip_high
        self.register_buffers("_steps", tc.tensor(0))
        self.register_buffers("_mean", tc.zeros(data_shape, dtype=tc.float32))
        self.register_buffers("_stddev", tc.zeros(data_shape, dtype=tc.float32))

    @property
    def steps(self):
        return self._steps.item()

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return self._stddev

    def __call__(self, item, eps=1e-4):
        steps = self.steps+1

        mean = self.mean
        mean *= ((steps-1) / steps)
        mean += (1 / steps) * item

        var = tc.square(self.stddev)
        var *= ((steps-1) / steps)
        var += (1 / steps) * tc.square(item-mean)
        stddev = tc.sqrt(var + eps)

        self.register_buffer(self._steps, tc.tensor(steps))
        self.register_buffer(self._mean, mean)
        self.register_buffer(self._stddev, stddev)

        normalized = (item - mean) / stddev
        if self._clip_low is not None:
            lows = tc.ones_like(normalized) * self._clip_low
            normalized = tc.max(lows, normalized)
        if self._clip_high is not None:
            highs = tc.ones_like(normalized) * self._clip_high
            normalized = tc.min(normalized, highs)
        return normalized


class NormalizeObservationsWrapper(StatefulWrapper, ObservationWrapper):
    def __init__(self, env, clip_low, clip_high):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            clip_low (Optional[float]): Minimum value after normalization.
            clip_high (Optional[float]): Maximum value after normalization.
        """
        super().__init__(env)
        self._data_shape = env.observation_space.shape
        self._clip_low = clip_low
        self._clip_high = clip_high
        self._normalizer = Normalizer(self._data_shape, clip_low, clip_high)

    def observation(self, observation):
        return self._normalizer.step(observation)

    def get_checkpointables(self):
        return {"observation_normalizer": self._normalizer}


class NormalizeRewardsWrapper(StatefulWrapper, RewardWrapper):
    def __init__(self, env, clip_low, clip_high):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            clip_low (Optional[float]): Minimum value after normalization.
            clip_high (Optional[float]): Maximum value after normalization.
        """
        super().__init__(env)
        self._data_shape = (1,)
        self._clip_low = clip_low
        self._clip_high = clip_high
        self._normalizer = Normalizer(self._data_shape, clip_low, clip_high)

    def reward(self, reward):
        return self._normalizer.step(tc.tensor(reward).unsqueeze(0)).item()

    def get_checkpointables(self):
        return {"reward_normalizer": self._normalizer}
