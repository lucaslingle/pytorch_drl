from drl.envs.wrappers.stateless.abstract import ActionWrapper


class StickyActionsWrapper(ActionWrapper):
    def __init__(self, env, stick_prob):
        super().__init__(env)
        self._stick_prob = stick_prob
        self._last_action = 0

    def action(self, action):
        u = self.unwrapped.np_random.uniform(low=0., high=1.)
        if u < self._stick_prob:
            return self._last_action
        return action

    def reverse_action(self, action):
        raise NotImplementedError

    def step(self, action):
        return self.env.step(self.action(action))
