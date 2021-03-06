"""
Checkpoint util.
"""

from typing import Optional, Dict, Union
import os
import re

import torch as tc

from drl.utils.types import Checkpointable


def format_name(kind: str, steps: int, suffix: str) -> str:
    filename = f"{kind}_{steps}.{suffix}"
    return filename


def parse_name(filename: str) -> Dict[str, Union[str, int]]:
    m = re.match(r"(\w+)_([0-9]+).([a-z]+)", filename)
    return {"kind": m.group(1), "steps": int(m.group(2)), "suffix": m.group(3)}


def _latest_n_checkpoint_steps(base_path, n=5, kind=''):
    ls = os.listdir(base_path)
    grep = [f for f in ls if parse_name(f)['kind'].startswith(kind)]
    steps = set(map(lambda f: parse_name(f)['steps'], grep))
    latest_steps = sorted(steps)
    latest_n = latest_steps[-n:]
    return latest_n


def _latest_step(base_path, kind=''):
    latest_steps = _latest_n_checkpoint_steps(base_path, n=1, kind=kind)
    return latest_steps[-1] if len(latest_steps) > 0 else None


def _clean(base_path, kind, n=5):
    latest_n_steps = _latest_n_checkpoint_steps(base_path, n=n, kind=kind)
    for fname in os.listdir(base_path):
        parsed = parse_name(fname)
        if parsed['kind'] == kind and parsed['steps'] not in latest_n_steps:
            os.remove(os.path.join(base_path, fname))


def _maybe_load_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Checkpointable,
        map_location: str,
        steps: Optional[int]) -> int:
    # loads a checkpoint if it exists, otherwise fails gracefully,
    # allowing training from scratch.
    base_path = checkpoint_dir
    os.makedirs(base_path, exist_ok=True)
    steps_ = _latest_step(base_path, kind_name) if steps is None else steps
    path = os.path.join(base_path, format_name(kind_name, steps_, 'pth'))
    if not os.path.exists(path):
        msg = f"Bad {kind_name} checkpoint or none at {base_path} with step {steps}."
        print(msg)
        print("Running from scratch.")
        return 0
    state_dict = tc.load(path, map_location=map_location)
    checkpointable.load_state_dict(state_dict)
    msg = f"Loaded {kind_name} checkpoint from {base_path}, with step {steps_}."
    print(msg)
    print("Continuing from checkpoint.")
    return steps_


def _save_checkpoint(
        checkpoint_dir: str,
        kind_name: str,
        checkpointable: Checkpointable,
        steps: int) -> None:
    # saves a checkpoint using name format provided in _format_name.
    base_path = checkpoint_dir
    os.makedirs(base_path, exist_ok=True)
    path = os.path.join(base_path, format_name(kind_name, steps, 'pth'))
    state_dict = checkpointable.state_dict()
    tc.save(state_dict, path)
    _clean(base_path, kind_name, n=5)


def maybe_load_checkpoints(
        checkpoint_dir: str,
        checkpointables: Dict[str, Optional[Checkpointable]],
        map_location: str,
        steps: Optional[int]) -> int:
    """
    Loads checkpoints from a provided checkpoint directory.

    Args:
        checkpoint_dir (str): Checkpoint directory.
        checkpointables (Dict[str, Optional[Checkpointable]]): Dictionary of
            checkpointables keyed by kind name.
        map_location (str): Map location specifying how remap storage locations.
        steps (Optional[int]): Number of steps so far. If None, uses latest.

    Returns:
        int: Number of steps in latest checkpoint. If no checkpoints, returns 0.
    """
    global_steps = list()
    for kind_name in checkpointables:
        checkpointable = checkpointables.get(kind_name)
        if checkpointable is not None:
            step_ = _maybe_load_checkpoint(
                checkpoint_dir=checkpoint_dir,
                kind_name=kind_name,
                checkpointable=checkpointable,
                map_location=map_location,
                steps=steps)
            global_steps.append(step_)
    if len(set(global_steps)) != 1:
        msg = "Checkpoint steps not aligned."
        raise RuntimeError(msg)
    return set(global_steps).pop()


def save_checkpoints(
        checkpoint_dir: str,
        checkpointables: Dict[str, Optional[Checkpointable]],
        steps: int) -> None:
    """
    Saves checkpoints to a provided checkpoint directory.

    Args:
        checkpoint_dir (str): Checkpoint directory.
        checkpointables (Dict[str, Optional[Checkpointable]]): Dictionary of
            checkpointables keyed by kind name.
        steps (int): Number of steps so far.

    Returns:
        None
    """
    for kind_name in checkpointables:
        checkpointable = checkpointables.get(kind_name)
        if checkpointable is not None:
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                kind_name=kind_name,
                checkpointable=checkpointable,
                steps=steps)
