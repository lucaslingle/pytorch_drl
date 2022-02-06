import torch as tc
import numpy as np

from drl.algos.common.grad_ops import (
    read_gradient,
    write_gradient,
    task_losses_to_grads,
    pcgrad_gradient_surgery,
    apply_pcgrad)
from drl.agents.architectures import NatureCNN
from drl.utils.initializers import get_initializer


def make_linear_model():
    model = tc.nn.Linear(3, 1, bias=False)
    tc.nn.init.zeros_(model.weight)
    return model


def make_big_model():
    model = NatureCNN(
        img_channels=4,
        w_init=get_initializer(('xavier_uniform_', {})),
        b_init=get_initializer(('zeros_', {})))
    return model


def make_optimizer(model, lr=0.01):
    optimizer = tc.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    return optimizer


def test_read_gradient():
    model = make_linear_model()
    lr = 0.01
    optimizer = make_optimizer(model, lr)

    features = tc.tensor([[3., 2., 1.]], requires_grad=False)
    labels = tc.tensor([[1.]], requires_grad=False)
    loss = tc.square(model(features) - labels).sum(dim=-1).mean(dim=0)
    loss.backward()

    grad_actual = tc.tensor(read_gradient(model, normalize=False))
    grad_expected = -2 * (1 - 0) * features.squeeze(0)
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

    model = make_linear_model()
    lr = 0.01
    optimizer = make_optimizer(model, lr)

    features = tc.tensor([[3., 2., 1.]], requires_grad=False)
    labels = tc.tensor([[1.]], requires_grad=False)
    loss = tc.square(model(features) - labels).sum(dim=-1).mean(dim=0)
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
    model = make_big_model()

    lr = 0.01
    optimizer = make_optimizer(model, lr)

    image_batch = tc.ones(size=[1, *model.input_shape], dtype=tc.float32)
    labels = tc.ones(size=[1, model.output_dim], dtype=tc.float32)
    loss = tc.square(model(image_batch) - labels).sum(dim=-1).mean(dim=0)
    loss.backward()

    g1 = read_gradient(model, normalize=False)
    optimizer.zero_grad()
    write_gradient(model, g1)
    g2 = read_gradient(model, normalize=False)

    tc.testing.assert_close(actual=tc.tensor(g2), expected=tc.tensor(g1))


def make_task_losses(model, features):
    task1_labels = tc.tensor([[0.]], requires_grad=False)
    task2_labels = tc.tensor([[1.]], requires_grad=False)
    model_pred = model(features)
    loss1 = tc.square(model_pred - task1_labels).sum(dim=-1).mean(dim=0)
    loss2 = tc.square(model_pred - task2_labels).sum(dim=-1).mean(dim=0)
    task_losses = {'task1': loss1, 'task2': loss2}
    return task_losses


def test_task_losses_to_grads():
    model = make_linear_model()
    optimizer = make_optimizer(model)

    features = tc.tensor([[3., 2., 1.]], requires_grad=False)
    task_losses = make_task_losses(model, features)
    task_grads_actual = task_losses_to_grads(
        network=model,
        optimizer=optimizer,
        task_losses=task_losses,
        normalize=False)
    task_grads_expected = {
        'task1': -2 * 0 * features, 'task2': -2 * 1 * features
    }
    for task in task_losses.keys():
        tc.testing.assert_close(
            actual=tc.tensor(task_grads_actual[task]).unsqueeze(0).float(),
            expected=task_grads_expected[task])


def test_pcgrad_gradient_surgery():
    task_gradients = {
        'task1': np.array([0., 0., 1.]), 'task2': np.array([1., 0., 0.])
    }
    pcgrad_output_expected = tc.tensor([1., 0., 1.]).float()
    pcgrad_output_actual = tc.tensor(
        pcgrad_gradient_surgery(task_gradients)).float()
    tc.testing.assert_close(
        actual=pcgrad_output_actual, expected=pcgrad_output_expected)

    task_gradients = {
        'task1': np.array([0., 0., 1.]), 'task2': np.array([2., 0., -2.])
    }
    pcgrad_output_expected = tc.tensor([2.5, 0., 0.5]).float()
    pcgrad_output_actual = tc.tensor(
        pcgrad_gradient_surgery(task_gradients)).float()
    tc.testing.assert_close(
        actual=pcgrad_output_actual, expected=pcgrad_output_expected)


def test_apply_pcgrad():
    model = make_linear_model()
    optimizer = make_optimizer(model)

    task1_features = tc.tensor([0., 0., -0.5])
    task2_features = tc.tensor([-1., 0., 1])
    label = tc.tensor([[1]])
    task1_loss = tc.square(model(task1_features) -
                           label).sum(dim=-1).mean(dim=0)
    task2_loss = tc.square(model(task2_features) -
                           label).sum(dim=-1).mean(dim=0)
    task_losses = {'task1': task1_loss, 'task2': task2_loss}
    #task_gradients = {
    #    'task1': np.array([0., 0., 1.]),
    #    'task2': np.array([2., 0., -2.])
    #}
    gradient_expected = tc.tensor([2.5, 0., 0.5]).float()

    apply_pcgrad(model, optimizer, task_losses, normalize=False)
    gradient_actual = tc.tensor(read_gradient(model, normalize=False)).float()

    tc.testing.assert_close(actual=gradient_actual, expected=gradient_expected)
