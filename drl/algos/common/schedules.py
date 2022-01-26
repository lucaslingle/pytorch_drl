class LinearSchedule:
    """
    Schedule for linearly annealing something.
    """
    def __init__(self, initial_value, final_value, final_step):
        self._initial_value = initial_value
        self._final_value = final_value
        self._final_step = final_step

    def value(self, global_step):
        frac_done = global_step / self._final_step
        s = min(max(0., frac_done), 1.)
        return self._initial_value * (1. - s) + self._final_value * s
