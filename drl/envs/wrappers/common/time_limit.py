from typing import Optional

from drl.envs.wrappers.common.abstract import Wrapper


class TimeLimitWrapper(Wrapper):
    def __init__(self, env, max_episode_steps):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            max_episode_steps (int): Maximum number of env steps.
        """
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        if self._elapsed_steps is None:
            msg = "Cannot call env.step() before calling reset()"
            raise RuntimeError(msg)
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, seed: Optional[int] = None, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, **kwargs)
