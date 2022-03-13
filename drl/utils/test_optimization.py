import torch as tc

from drl.utils.optimization import (
    get_optimizer, get_scheduler, get_weight_decay_param_groups)


def test_get_optimizer():
    model = tc.nn.Linear(10, 3)

    optimizer = get_optimizer(
        model, cls_name='Adam', cls_args={
            'lr': 0.001, 'betas': [0.5, 0.999]
        })
    assert isinstance(optimizer, tc.optim.Adam)
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 0.001
        assert param_group['betas'] == [0.5, 0.999]

    optimizer = get_optimizer(
        model, cls_name='AdamW', cls_args={
            'lr': 0.001, 'wd': 0.01
        })
    assert isinstance(optimizer, tc.optim.AdamW)
    print(optimizer.param_groups)
    for param_group in optimizer.param_groups:
        assert param_group['lr'] == 0.001
        assert param_group['betas'] == (0.9, 0.999)
        for param in param_group['params']:
            assert isinstance(param, tc.nn.Parameter)
            if len(param.shape) > 1:
                assert param_group['weight_decay'] == 0.01
            else:
                assert param_group['weight_decay'] == 0.0


def test_get_scheduler():
    model = tc.nn.Linear(10, 3)
    optimizer = tc.optim.SGD(model.parameters(), lr=0.01)

    scheduler = get_scheduler(optimizer, cls_name='None', cls_args={})
    assert scheduler is None

    scheduler = get_scheduler(optimizer, cls_name='None', cls_args={'foo': 1.0})
    assert scheduler is None

    scheduler = get_scheduler(
        optimizer,
        cls_name='OneCycleLR',
        cls_args={
            'max_lr': 0.001,
            'total_steps': 10000,
            'pct_start': 0.0,
            'anneal_strategy': 'linear',
            'cycle_momentum': False,
            'base_momentum': 0.0,
            'max_momentum': 0.0,
            'div_factor': 1.0
        })
    assert isinstance(scheduler, tc.optim.lr_scheduler.OneCycleLR)


def test_get_weight_decay_param_groups():
    model = tc.nn.Linear(10, 3)
    param_groups = get_weight_decay_param_groups(model, wd=0.01)
    for param_group in param_groups:
        for param in param_group['params']:
            assert isinstance(param, tc.nn.Parameter)
            if len(param.shape) > 1:
                assert param_group['weight_decay'] == 0.01
            else:
                assert param_group['weight_decay'] == 0.0
