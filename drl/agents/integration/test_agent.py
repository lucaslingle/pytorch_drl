import torch as tc

from drl.agents.preprocessing import OneHotEncode
from drl.agents.architectures import Identity, Linear
from drl.utils.initializers import get_initializer
from drl.agents.heads import CategoricalPolicyHead, SimpleValueHead
from drl.agents.integration import Agent


def test_agent_tabular():
    batch_size = 1
    num_states = 3
    num_actions = 2
    preprocessings = [
        OneHotEncode(depth=num_states)
    ]
    architecture = Identity(
        input_shape=[num_states],
        w_init=get_initializer(('zeros_', {})),
        b_init=get_initializer(('zeros_', {}))
    )
    heads = {
        'policy': CategoricalPolicyHead(
            num_features=num_states,
            num_actions=num_actions,
            head_architecture_cls=Linear,
            head_architecture_cls_args={},
            w_init=get_initializer(('zeros_', {})),
            b_init=get_initializer(('zeros_', {}))
        ),
        'value_extrinsic': SimpleValueHead(
            num_features=num_states,
            head_architecture_cls=Linear,
            head_architecture_cls_args={},
            w_init=get_initializer(('zeros_', {})),
            b_init=get_initializer(('zeros_', {}))
        )
    }

    tabular_agent = Agent(
        preprocessing=preprocessings,
        architecture=architecture,
        predictors=heads
    )
    observation_batch = tc.zeros(size=(batch_size,), dtype=tc.int64)
    predictions = tabular_agent(
        observations=observation_batch,
        predict=['policy', 'value_extrinsic'])
    print(predictions['value_extrinsic'])

    tc.testing.assert_close(
        actual=predictions['value_extrinsic'],
        expected=tc.zeros(size=[batch_size], dtype=tc.float32),
        rtol=1e-4,
        atol=1e-4
    )
    tc.testing.assert_close(
        actual=predictions['policy'].log_prob(
            tc.zeros(size=[batch_size], dtype=tc.int64)
        ),
        expected=tc.log(
            (1. / num_actions) * tc.ones(size=[batch_size], dtype=tc.float32)
        ),
        rtol=1e-4,
        atol=1e-4
    )
