import abc

import torch as tc


class CategoricalPolicyHeadMixin(tc.nn.Module, metaclass=abc.ABCMeta):
    def policy(self, x):
        features = self.forward(x)
        logits = self.__policy_head(features)
        dist = tc.distributions.Categorical(logits=logits)
        return dist


class DiagonalGaussianPolicyHeadMixin(tc.nn.Module, metaclass=abc.ABCMeta):
    def policy(self, x):
        features = self.forward(x)
        vec = self.__policy_head(features)
        mu, logsigma = tc.chunk(vec, 2, dim=-1)
        dist = tc.distributions.Normal(loc=mu, scale=tc.exp(logsigma))
        return dist


class ValueHeadMixin(tc.nn.Module, metaclass=abc.ABCMeta):
    def value(self, x):
        features = self.forward(x)
        vpred = self.__value_head(features).squeeze(-1)
        return vpred
