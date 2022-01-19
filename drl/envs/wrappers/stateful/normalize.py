import torch as tc


class Normalizer(tc.nn.Module):
    # todo(lucaslingle):
    #      consider using this algorithm instead
    #      https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, data_shape, clip_low=None, clip_high=None):
        super().__init__()
        self._data_shape = data_shape
        self._clip_low = clip_low
        self._clip_high = clip_high
        self.register_buffers("_steps", tc.tensor(0.))
        self.register_buffers("_moment1", tc.zeros(data_shape, dtype=tc.float32))
        self.register_buffers("_moment2", tc.zeros(data_shape, dtype=tc.float32))

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, value):
        self.register_buffer(self._steps, value)

    @property
    def moment1(self):
        return self._moment1

    @moment1.setter
    def moment1(self, tensor):
        self.register_buffer('_moment1', tensor)

    @property
    def moment2(self):
        return self._moment2

    @moment2.setter
    def moment2(self, tensor):
        self.register_buffer('_moment2', tensor)

    def update(self, item):
        # updates a streaming, asymptotically-unbiased estimator of mean and var
        steps = self.steps + 1

        moment1 = self.moment1
        moment1 *= ((steps-1) / steps)
        moment1 += (1 / steps) * item

        moment2 = self.moment2
        moment2 *= ((steps-1) / steps)
        moment2 += (1 / steps) * tc.square(item)

        self.steps = steps
        self.moment1 = moment1
        self.moment2 = moment2

    def forward(self, item, shift=True, scale=True, eps=1e-4):
        m1, m2 = self.moment1.unsqueeze(0), self.moment2.unsqueeze(0)
        mean = m1
        var = m2 - tc.square(m1)
        if shift:
            item -= mean
        if scale:
            item *= tc.rsqrt(var + eps)
        if self._clip_low is not None:
            lows = tc.ones_like(item) * self._clip_low
            item = tc.max(lows, item)
        if self._clip_high is not None:
            highs = tc.ones_like(item) * self._clip_high
            item = tc.min(item, highs)
        return item
