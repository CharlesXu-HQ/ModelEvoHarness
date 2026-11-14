"""Independent formula checks prompted by the rec_models source comparison.

These tests use explicit scalar/loop oracles, not third-party model wrappers.
Variant differences (CIN activation, PLE shared routing) are intentional.
"""
import importlib

import numpy as np
import pytest


def _linear(layer, weight, bias, framework):
    if framework == "pytorch":
        import torch
        with torch.no_grad():
            layer.weight.copy_(torch.tensor(weight).T)
            layer.bias.copy_(torch.tensor(bias))
    else:
        layer.set_weights([np.asarray(weight, np.float32), np.asarray(bias, np.float32)])


def _tensor(values, framework):
    module = pytest.importorskip("torch" if framework == "pytorch" else "tensorflow")
    return module.tensor(values, dtype=module.float32) if framework == "pytorch" else module.constant(values, dtype=module.float32)


def _numpy(value, framework):
    return value.detach().numpy() if framework == "pytorch" else value.numpy()


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_mmoe_has_task_specific_normalized_expert_mixtures(framework):
    x = _tensor([[1., 2.], [2., 1.]], framework)
    cls = getattr(importlib.import_module(f"model_evo_harness.models.{framework}.architectures"), "MMoE")
    model = cls(2, 2, 2, 1)
    model(x)
    for i, expert in enumerate(model.experts):
        dense = expert[0] if framework == "pytorch" else expert
        _linear(dense, [[i + 1.], [2. - i]], [.5], framework)
    for i, (gate, head) in enumerate(zip(model.gates, model.heads)):
        _linear(gate, [[1., -1.], [0., 0.]], [i, -i], framework)
        _linear(head, [[1.]], [0.], framework)
    expected = []
    for row in np.asarray([[1., 2.], [2., 1.]]):
        values = np.asarray([row[0] + 2 * row[1] + .5, 2 * row[0] + row[1] + .5])
        task_outputs = []
        for task in range(2):
            logits = np.asarray([row[0] + task, -row[0] - task])
            weights = np.exp(logits - logits.max()); weights /= weights.sum()
            task_outputs.append(weights @ values)
        expected.append(task_outputs)
    np.testing.assert_allclose(_numpy(model(x), framework), expected, rtol=1e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_ple_shared_stream_receives_other_task_experts_and_gradients(framework):
    x = _tensor([[1.], [2.]], framework)
    cls = getattr(importlib.import_module(f"model_evo_harness.models.{framework}.multitask"), "PLE")
    model = cls(1, 2, 1, 1, 1, levels=2)
    model(x)
    for level in range(2):
        groups = [model.shared_experts[level][0], model.task_experts[level][0][0],
                  model.task_experts[level][1][0]]
        for expert, slope in zip(groups, (2., 3., 5.) if level == 0 else (1., 1., 1.)):
            dense = expert[0] if framework == "pytorch" else expert
            _linear(dense, [[slope]], [0.], framework)
        for gate in model.task_gates[level]:
            _linear(gate, [[0., 0.]], [0., 0.], framework)
    _linear(model.shared_gates[0], [[0., 0., 0.]], [0., 0., 0.], framework)
    for head in model.heads:
        _linear(head, [[1.]], [0.], framework)
    # First shared state: (2+3+5)x/3. Task states: (3+2)x/2, (5+2)x/2.
    expected = np.asarray([[35/12, 41/12], [35/6, 41/6]])
    if framework == "pytorch":
        output = model(x)
        output[:, 0].sum().backward()
        cross_task_gradient = model.task_experts[0][1][0][0].weight.grad
    else:
        import tensorflow as tf
        with tf.GradientTape() as tape:
            output = model(x)
            loss = tf.reduce_sum(output[:, 0])
        cross_task_gradient = tape.gradient(loss, model.task_experts[0][1][0].kernel)
    np.testing.assert_allclose(_numpy(output, framework), expected, rtol=1e-6)
    # Other task's first expert reaches task 0 through the intermediate shared gate.
    np.testing.assert_allclose(_numpy(cross_task_gradient, framework), [[.5]], rtol=1e-6)


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_cin_field_pair_order_and_relu_variant_match_scalar_oracle(framework):
    module = pytest.importorskip("torch" if framework == "pytorch" else "tensorflow")
    cls = getattr(importlib.import_module(f"model_evo_harness.models.{framework}.interactions"), "XDeepFM")
    model = cls([2, 2], 2, hidden_dim=2, cin_channels=[2, 1])
    ids = module.tensor([[0, 1], [1, 0]]) if framework == "pytorch" else module.constant([[0, 1], [1, 0]])
    table = np.asarray([[1., -2.], [-1., 3.], [2., .5], [-2., 1.]], np.float32)
    weights = [np.asarray([[1., -.5, .2, -1.], [-.3, .5, 1., .2]], np.float32),
               np.asarray([[.2, -.5, 1., .7]], np.float32)]
    if framework == "pytorch":
        with module.no_grad():
            model.fields.embeddings.weight.copy_(module.tensor(table))
            for parameter, value in zip(model.cin_weights, weights):
                parameter.copy_(module.tensor(value))
    else:
        model.fields.embeddings.set_weights([table])
        for parameter, value in zip(model.cin_weights, weights):
            parameter.assign(value)
    original = table[np.asarray([[0, 3], [1, 2]])]
    hidden, readouts = original, []
    negative_seen = False
    for weight in weights:
        output = np.zeros((2, len(weight), 2))
        for batch in range(2):
            for channel in range(len(weight)):
                for coordinate in range(2):
                    score = sum(weight[channel, i * hidden.shape[1] + j] *
                                original[batch, i, coordinate] * hidden[batch, j, coordinate]
                                for i in range(2) for j in range(hidden.shape[1]))
                    negative_seen |= score < 0
                    output[batch, channel, coordinate] = max(score, 0.)
        hidden = output
        readouts.append(output.sum(axis=-1))
    assert negative_seen  # Distinguishes the declared ReLU variant from a linear CIN.
    np.testing.assert_allclose(_numpy(model.cin_features(ids), framework),
                               np.concatenate(readouts, axis=1), rtol=1e-6, atol=1e-6)
