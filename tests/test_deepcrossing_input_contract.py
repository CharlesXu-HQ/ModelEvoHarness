"""DeepCrossing must honor the declared feature width on its first call."""

import numpy as np
import pytest


@pytest.mark.parametrize("actual_width", [3, 5])
def test_tensorflow_deepcrossing_rejects_wrong_width_before_build(actual_width):
    tf = pytest.importorskip("tensorflow")
    from model_evo_harness.models.tensorflow.paper_ranking import DeepCrossing

    model = DeepCrossing(input_dim=4, hidden_dim=2)
    with pytest.raises(ValueError):
        model(tf.ones((2, actual_width)))


def test_tensorflow_deepcrossing_accepts_declared_width_and_keeps_gradients():
    tf = pytest.importorskip("tensorflow")
    from model_evo_harness.models.tensorflow.paper_ranking import DeepCrossing

    model = DeepCrossing(input_dim=4, hidden_dim=2, num_blocks=1)
    model(tf.ones((1, 4)))
    # Keep the ReLU paths active so the input-gradient assertion is deterministic.
    for variable in model.trainable_variables:
        variable.assign(tf.fill(variable.shape, .1))
    for batch_size in (1, 3):
        features = tf.Variable(tf.ones((batch_size, 4)))
        with tf.GradientTape() as tape:
            logits = model(features)
            loss = tf.reduce_sum(logits)
        gradients = tape.gradient(loss, [features] + model.trainable_variables)
        assert logits.shape == (batch_size,)
        assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
        assert np.all(gradients[0].numpy() > 0)
