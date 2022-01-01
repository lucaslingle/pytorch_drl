from drl.envs.wrappers.common.abstract import Wrapper
from drl.utils.typing_util import Env


class EpisodicLifeWrapper(Wrapper):
    """
    Use loss-of-life to mark end-of-episode using the done indicator.
    """
    def __init__(self, env, lives_fn, noop_action):
        """
        Args:
            env (Env): OpenAI gym environment instance.
            lives_fn (Callable[[Env], int]): Function to obtain num lives.
            noop_action (int): Any no-op action.
        """
        super().__init__(env)
        self._was_real_done = True
        self._lives = 0
        self._lives_fn = lives_fn
        self._noop_action = noop_action

    def step(self, action):
        obs, reward, self._was_real_done, info = self.env.step(action)

        lives = self._lives_fn(self.env)
        done = self.was_real_done or (0 < lives < self._lives)
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        Resets only when lives are exhausted.
        """
        if self._was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, *_ = self.env.step(self._noop_action)
        self._lives = self._lives_fn(self.env)
        return obs
