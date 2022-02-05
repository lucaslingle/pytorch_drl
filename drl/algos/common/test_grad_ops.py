import torch as tc

from drl.algos.common.grad_ops import read_gradient, write_gradient
from drl.agents.architectures import NatureCNN
from drl.utils.initializers import get_initializer


def test_read_gradient():
    model = tc.nn.Linear(3, 1, bias=False)
    tc.nn.init.zeros_(model.weight)

    lr = 0.01
    optimizer = tc.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    criterion = tc.nn.MSELoss(reduction='mean')

    features = tc.tensor([[3., 2., 1.]], requires_grad=False)
    labels = tc.tensor([[1.]], requires_grad=False)
    loss = criterion(
        input=model(features).squeeze(-1), target=labels.squeeze(-1))
    loss.backward()

    grad_actual = tc.tensor(read_gradient(model, normalize=False))
    grad_expected = -2 * features.squeeze(0)
    tc.testing.assert_close(actual=grad_actual, expected=grad_expected)

    optimizer.step()
    tc.testing.assert_close(
        actual=list(model.parameters())[0],
        expected=-lr * grad_actual.unsqueeze(0))  # output dim first in pytorch


def test_write_gradient():
    # (1) reads a gradient from a model with the same setup as in the
    # test_read_gradient test.
    # (2) then zeros out the grad field of the params, writes it back,
    # (3) steps the params
    # (4) check if the results match what they would've been
    # if we hadn't zeroed and rewritten the gradient back
    # to the params' grad field.

    model = tc.nn.Linear(3, 1, bias=False)
    tc.nn.init.zeros_(model.weight)

    lr = 0.01
    optimizer = tc.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    criterion = tc.nn.MSELoss(reduction='mean')

    features = tc.tensor([[3., 2., 1.]], requires_grad=False)
    labels = tc.tensor([[1.]], requires_grad=False)
    loss = criterion(
        input=model(features).squeeze(-1), target=labels.squeeze(-1))
    loss.backward()

    g = read_gradient(model, normalize=False)
    optimizer.zero_grad()

    write_gradient(model, g)
    optimizer.step()
    tc.testing.assert_close(
        actual=list(model.parameters())[0],
        expected=-lr * tc.tensor(g).unsqueeze(0))  # output dim first in pytorch


def test_write_gradient_bigmodel():
    # reads gradients, zeros em, writes em back and reads em back.
    # checks if the two reads match.
    model = NatureCNN(
        img_channels=4,
        w_init=get_initializer(('xavier_uniform_', {})),
        b_init=get_initializer(('zeros_', {})))

    lr = 0.01
    optimizer = tc.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()

    image_batch = tc.ones(size=[1, *model.input_shape], dtype=tc.float32)
    labels = tc.ones(size=[1, model.output_dim], dtype=tc.float32)
    loss = tc.square(model(image_batch) - labels).sum(dim=-1).mean(dim=0)
    loss.backward()

    g1 = read_gradient(model, normalize=False)
    optimizer.zero_grad()
    write_gradient(model, g1)
    g2 = read_gradient(model, normalize=False)

    tc.testing.assert_close(actual=tc.tensor(g1), expected=tc.tensor(g2))
