from drl.envs.wrappers.abstract import Wrapper


ATARI_LIFES_FN = lambda env: env.unwrapped.ale.lives()


class EpisodicLifeWrapper(Wrapper):
    """
    Use loss-of-life to mark end-of-episode using the done indicator.
    """
    def __init__(self, env, lives_fn):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param lives_fn (Callable[[Env], int]): Function to obtain num lives.
        """
        super().__init__(self, env)
        self._was_real_done = True
        self._lives = 0
        self._lives_fn = lives_fn

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
            obs, *_ = self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        return obs
