from typing import Union, Mapping

import torch as tc
import gym

from drl.envs.wrappers import Wrapper


def update_trainable_wrappers(
        env: Union[gym.core.Env, Wrapper],
        mb: Mapping[str, tc.Tensor]
) -> None:
    """
    Updates/trains wrappers with state.

    Args:
        env (Union[gym.core.Env, Wrapper]): OpenAI gym env or Wrapper instance.
        mb (Mapping[str, tc.Tensor]): Minibatch of experience to train on.

    Returns:
        None.
    """
    maybe_wrapper = env
    while isinstance(maybe_wrapper, Wrapper):
        if hasattr(maybe_wrapper, 'learn'):
            maybe_wrapper.learn(mb)
        maybe_wrapper = maybe_wrapper.env
