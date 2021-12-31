from drl.envs.wrappers.common.abstract import Wrapper


ATARI_ACTION_SEQUENCE = [1, 2]


class ActionResetWrapper(Wrapper):
    """
    Take action on reset for environments that are fixed until
    action sequence occurs.
    """
    def __init__(self, env, action_sequence):
        """
        :param env (gym.core.Env): OpenAI gym environment instance.
        :param action_sequence (List[int]): List of actions to perform.
        """
        super().__init__(env)
        self._action_sequence = action_sequence

    def reset(self, **kwargs):
        """
        Resets environment and takes actions listed in self._action_sequence.
        """
        obs = self.env.reset(**kwargs)
        for a in self._action_sequence:
            obs, _, done, _ = self.env.step(a)
            if done:
                _ = self.env.reset(**kwargs)
        return obs
