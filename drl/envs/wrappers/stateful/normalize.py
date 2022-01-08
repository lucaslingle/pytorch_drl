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
        self.register_buffers("_mean", tc.zeros(data_shape, dtype=tc.float32))
        self.register_buffers("_var", tc.zeros(data_shape, dtype=tc.float32))

    @property
    def steps(self):
        return self._steps

    @steps.setter
    def steps(self, value):
        self.register_buffer(self._steps, value)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, tensor):
        self.register_buffer(self._mean, tensor)

    @property
    def var(self):
        return self._var

    @var.setter
    def var(self, tensor):
        self.register_buffer(self._var, tensor)

    def update(self, item):
        # updates a streaming, asymptotically-unbiased estimator of mean and var
        steps = self.steps + 1

        mean = self.mean
        mean *= ((steps-1) / steps)
        mean += (1 / steps) * item

        var = self.var
        var *= ((steps-1) / steps)
        var += (1 / steps) * tc.square(item-mean)

        self.steps = steps
        self.mean = mean
        self.var = var

    def forward(self, item, shift=True, scale=True, eps=1e-4):
        mean, var = self.mean.unsqueeze(0), self.var.unsqueeze(0)
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
