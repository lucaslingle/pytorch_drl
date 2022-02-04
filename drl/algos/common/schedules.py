class LinearSchedule:
    """
    Schedule for linearly annealing something.
    """
    def __init__(
            self, initial_value: float, final_value: float, final_step: int):
        """
        Args:
            initial_value (float): Initial value to anneal from.
            final_value (float): Final value to anneal to.
            final_step (int): Endpoint for annealing. After this,
                `final_value` will be returned no matter how many steps occur.
        """
        self._initial_value = initial_value
        self._final_value = final_value
        self._final_step = final_step

    def value(self, global_step: int) -> float:
        """
        Args:
            global_step (int): Global step.

        Returns:
            float: Annealed value, guaranteed to be between
                initial_value and final_value.
        """
        frac_done = global_step / self._final_step
        s = min(max(0., frac_done), 1.)
        return self._initial_value * (1. - s) + self._final_value * s
