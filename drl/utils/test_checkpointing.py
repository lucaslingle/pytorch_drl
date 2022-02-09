import tempfile
import os

import torch as tc

from drl.utils.checkpointing import (
    format_name, parse_name, save_checkpoints, maybe_load_checkpoints)

CHECKPOINT_DIR = os.path.join(tempfile.gettempdir(), 'pytorch_drl_testing')
STEPS = 7357


def make_checkpointables():
    model = tc.nn.Linear(10, 2, bias=True)
    optimizer = tc.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = tc.optim.lr_scheduler.LinearLR(optimizer=optimizer)
    return {'model': model, 'optimizer': optimizer, 'schedular': scheduler}


def test_format_name():
    assert format_name(kind='note', steps=0, suffix='txt') == 'note_0.txt'
    assert format_name(kind='foo', steps=42, suffix='phd') == 'foo_42.phd'
    assert format_name(kind='model', steps=12000, suffix='pth') == \
           'model_12000.pth'
    assert format_name(kind='a_b_c', steps=123, suffix='xyz') == 'a_b_c_123.xyz'


def test_parse_name():
    assert parse_name('note_0.txt') == {
        'kind': 'note', 'steps': 0, 'suffix': 'txt'
    }
    assert parse_name('foo_42.phd') == {
        'kind': 'foo', 'steps': 42, 'suffix': 'phd'
    }
    assert parse_name('model_12000.pth') == {
        'kind': 'model', 'steps': 12000, 'suffix': 'pth'
    }
    assert parse_name('a_b_c_123.xyz') == {
        'kind': 'a_b_c', 'steps': 123, 'suffix': 'xyz'
    }


def test_save_checkpoints():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpointables = make_checkpointables()
    save_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpointables=checkpointables,
        steps=STEPS)

    checkpointables_reloaded = make_checkpointables()
    for name in checkpointables:
        path = os.path.join(
            CHECKPOINT_DIR, format_name(kind=name, steps=STEPS, suffix='pth'))
        state_dict = tc.load(path, map_location='cpu')
        checkpointables_reloaded[name].load_state_dict(state_dict)

    for name in checkpointables:
        assert set(checkpointables[name].state_dict().keys()) == \
               set(checkpointables_reloaded[name].state_dict().keys())
        for key in checkpointables[name].state_dict().keys():
            tc.testing.assert_close(
                actual=checkpointables_reloaded[name].state_dict()[key],
                expected=checkpointables[name].state_dict()[key])


def test_maybe_load_checkpoints():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpointables = make_checkpointables()
    save_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpointables=checkpointables,
        steps=STEPS)

    checkpointables_reloaded = make_checkpointables()
    global_step = maybe_load_checkpoints(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpointables=checkpointables_reloaded,
        map_location='cpu',
        steps=STEPS)

    assert global_step == STEPS

    for name in checkpointables:
        assert set(checkpointables[name].state_dict().keys()) == \
               set(checkpointables_reloaded[name].state_dict().keys())
        for key in checkpointables[name].state_dict().keys():
            tc.testing.assert_close(
                actual=checkpointables_reloaded[name].state_dict()[key],
                expected=checkpointables[name].state_dict()[key])
