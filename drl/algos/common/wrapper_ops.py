from drl.envs.wrappers import Wrapper


def update_trainable_wrappers(env, mb):
    maybe_wrapper = env
    while isinstance(maybe_wrapper, Wrapper):
        if hasattr(maybe_wrapper, 'learn'):
            maybe_wrapper.learn(**mb)
        maybe_wrapper = maybe_wrapper.env
