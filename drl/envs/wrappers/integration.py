from typing import Dict, Any
import importlib

from drl.envs.wrappers.common.abstract import Wrapper


def get_wrapper(env, cls_name, cls_args):
    module = importlib.import_module('drl.envs.wrappers')
    cls = getattr(module, cls_name)
    return cls(env, **cls_args)


def get_wrappers(env, **wrappers_spec: Dict[str, Dict[str, Any]]) -> Wrapper:
    for cls_name, cls_arg in wrappers_spec.items():
        env = get_wrapper(env, cls_name, cls_arg)
    return env
