"""Contracts and distinctive mechanisms for retrieval references."""

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class PyTorchRetrievalTests(unittest.TestCase):
    def test_latent_factors_and_bias(self):
        import torch
        from model_evo_harness.models.pytorch.retrieval import BiasSVD, FunkSVD

        users = torch.tensor([0, 1, 2])
        items = torch.tensor([1, 2, 3])
        for model in (FunkSVD(3, 4, 5), BiasSVD(3, 4, 5)):
            with self.subTest(model=type(model).__name__):
                logits = model(users, items)
                self.assertEqual(tuple(logits.shape), (3,))
                logits.square().mean().backward()
                self.assertTrue(any(p.grad is not None for p in model.parameters()))
        biased = BiasSVD(3, 4, 5)
        before = biased(users, items).detach()
        with torch.no_grad():
            biased.user_bias.weight.fill_(2)
        self.assertTrue(torch.allclose(biased(users, items), before + 2))

    def test_positive_only_neighborhoods(self):
        import torch
        from model_evo_harness.models.pytorch.retrieval import ItemCF, Swing, UserCF

        interactions = torch.tensor([[1, 1, 0], [1, 1, 0],
                                     [0, 1, 1], [0, 0, 1]], dtype=torch.float32)
        item = ItemCF().fit(interactions)
        user = UserCF().fit(interactions)
        swing = Swing().fit(interactions)
        self.assertGreater(item.similarity[0, 1].item(), 0)
        self.assertEqual(item.similarity[0, 2].item(), 0)
        self.assertGreater(user.similarity[0, 1].item(), 0)
        self.assertEqual(user.similarity[0, 3].item(), 0)
        self.assertGreater(swing.similarity[0, 1].item(), 0)
        self.assertEqual(swing.similarity[1, 2].item(), 0)
        self.assertEqual(tuple(item.scores(interactions[:2]).shape), (2, 3))
        self.assertEqual(tuple(user.scores(torch.tensor([0, 2])).shape), (2, 3))
        self.assertEqual(tuple(swing.scores(interactions[:2]).shape), (2, 3))

    def test_side_features_and_skipgram(self):
        import torch
        from model_evo_harness.models.pytorch.retrieval import EGES, Item2Vec

        item_ids = torch.tensor([0, 1, 2])
        side_ids = torch.tensor([[0, 1], [1, 2], [0, 2]])
        eges = EGES(4, [3, 4], 6)
        weights = eges.attention_weights(item_ids)
        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(3)))
        self.assertFalse(torch.allclose(eges.encode_items(item_ids, side_ids),
                                        eges.encode_items(item_ids, side_ids.flip(0))))
        eges(item_ids, side_ids, torch.tensor([1, 2, 3])).sum().backward()
        self.assertIsNotNone(eges.item_embedding.weight.grad)

        skipgram = Item2Vec(5, 6)
        loss = skipgram.negative_sampling_loss(
            torch.tensor([0, 1]), torch.tensor([1, 2]),
            torch.tensor([[3, 4], [0, 4]]))
        self.assertEqual(tuple(loss.shape), ())
        loss.backward()
        self.assertIsNotNone(skipgram.input_embedding.weight.grad)
        self.assertEqual(tuple(skipgram.user_vector(torch.tensor([[0, 1]]),
                                                     torch.tensor([[1, 0]])).shape), (1, 6))

    def test_towers_and_sampling_correction(self):
        import torch
        from model_evo_harness.models.pytorch.retrieval import (
            DSSM, FMRecall, YouTubeDNN, YouTubeSBC)

        user_features = torch.randn(3, 5)
        item_features = torch.randn(3, 4)
        dssm = DSSM(5, 4, 6, 8)
        self.assertTrue(torch.allclose(dssm.encode_user(user_features).norm(dim=1),
                                        torch.ones(3), atol=1e-5))
        dssm(user_features, item_features).sum().backward()
        self.assertIsNotNone(dssm.user_tower[0].weight.grad)

        recall = FMRecall([4, 5], [6, 7], 6)
        user_fields = torch.tensor([[0, 1], [1, 2], [2, 3]])
        item_fields = torch.tensor([[1, 2], [2, 3], [3, 4]])
        scores = recall(user_fields, item_fields)
        expected = (recall.encode_user(user_fields) * recall.encode_item(item_fields)).sum(1)
        self.assertTrue(torch.allclose(scores, expected))
        scores.sum().backward()
        self.assertIsNotNone(recall.user_embeddings[0].weight.grad)

        candidates = torch.tensor([0, 1, 2, 3])
        history = torch.tensor([[0, 1], [1, 2], [2, 3]])
        history_mask = torch.tensor([[1, 1], [1, 0], [0, 0]])
        base = YouTubeDNN(5, 5, 6, 8)
        corrected = YouTubeSBC(5, 5, 6, 8)
        base_logits = base(user_features, candidates, history, history_mask)
        corrected(user_features, candidates, torch.ones(4), history, history_mask)
        corrected.load_state_dict(base.state_dict())
        probabilities = torch.tensor([0.5, 0.25, 0.2, 0.05])
        corrected_logits = corrected(user_features, candidates, probabilities,
                                      history, history_mask)
        self.assertTrue(torch.allclose(corrected_logits - base_logits,
                                        -probabilities.log()[None, :], atol=1e-5))
        self.assertEqual(tuple(base_logits.shape), (3, 4))
        corrected_logits.sum().backward()
        self.assertIsNotNone(corrected.item_embedding.weight.grad)


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
class TensorFlowRetrievalTests(unittest.TestCase):
    def test_neighborhood_mechanisms(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.retrieval import ItemCF, Swing, UserCF

        interactions = tf.constant([[1, 1, 0], [1, 1, 0],
                                    [0, 1, 1], [0, 0, 1]], dtype=tf.float32)
        item = ItemCF().fit(interactions)
        user = UserCF().fit(interactions)
        swing = Swing().fit(interactions)
        self.assertGreater(float(item.similarity[0, 1]), 0)
        self.assertEqual(float(item.similarity[0, 2]), 0)
        self.assertGreater(float(user.similarity[0, 1]), 0)
        self.assertEqual(float(user.similarity[0, 3]), 0)
        self.assertGreater(float(swing.similarity[0, 1]), 0)
        self.assertEqual(float(swing.similarity[1, 2]), 0)
        self.assertEqual(tuple(item.scores(interactions[:2]).shape), (2, 3))
        self.assertEqual(tuple(user.scores(tf.constant([0, 2])).shape), (2, 3))

    def test_neural_shapes_gradients_and_correction(self):
        import tensorflow as tf
        from model_evo_harness.models.tensorflow.retrieval import (
            BiasSVD, DSSM, EGES, FMRecall, FunkSVD, Item2Vec, YouTubeDNN,
            YouTubeSBC)

        users = tf.constant([0, 1, 2])
        items = tf.constant([1, 2, 3])
        user_features = tf.random.normal((3, 5))
        item_features = tf.random.normal((3, 4))
        side_ids = tf.constant([[0, 1], [1, 2], [0, 2]])
        user_fields = tf.constant([[0, 1], [1, 2], [2, 3]])
        item_fields = tf.constant([[1, 2], [2, 3], [3, 4]])
        candidates = tf.constant([0, 1, 2, 3])
        history = tf.constant([[0, 1], [1, 2], [2, 3]])
        history_mask = tf.constant([[1, 1], [1, 0], [0, 0]])
        models_and_calls = [
            (FunkSVD(3, 4, 5), lambda m: m(users, items), (3,)),
            (BiasSVD(3, 4, 5), lambda m: m(users, items), (3,)),
            (EGES(4, [3, 4], 6), lambda m: m(items - 1, side_ids, items), (3,)),
            (DSSM(5, 4, 6, 8), lambda m: m(user_features, item_features), (3,)),
            (FMRecall([4, 5], [6, 7], 6),
             lambda m: m(user_fields, item_fields), (3,)),
            (YouTubeDNN(5, 5, 6, 8),
             lambda m: m(user_features, candidates, history, history_mask), (3, 4)),
            (YouTubeSBC(5, 5, 6, 8),
             lambda m: m(user_features, candidates, tf.ones(4), history,
                         history_mask), (3, 4)),
        ]
        for model, call, shape in models_and_calls:
            with self.subTest(model=type(model).__name__):
                with tf.GradientTape() as tape:
                    scores = call(model)
                    loss = tf.reduce_sum(tf.square(scores))
                self.assertEqual(tuple(scores.shape), shape)
                grads = tape.gradient(loss, model.trainable_variables)
                self.assertTrue(any(g is not None for g in grads))

        skipgram = Item2Vec(5, 6)
        with tf.GradientTape() as tape:
            loss = skipgram.negative_sampling_loss(
                tf.constant([0, 1]), tf.constant([1, 2]),
                tf.constant([[3, 4], [0, 4]]))
        self.assertIsNotNone(tape.gradient(loss, skipgram.trainable_variables)[0])

        base = YouTubeDNN(5, 5, 6, 8)
        corrected = YouTubeSBC(5, 5, 6, 8)
        logits = base(user_features, candidates, history, history_mask)
        corrected(user_features, candidates, tf.ones(4), history, history_mask)
        corrected.set_weights(base.get_weights())
        p = tf.constant([0.5, 0.25, 0.2, 0.05])
        difference = corrected(user_features, candidates, p, history,
                               history_mask) - logits
        self.assertTrue(bool(tf.reduce_all(tf.abs(difference + tf.math.log(p)[None, :])
                                           < 1e-5)))


if __name__ == "__main__":
    unittest.main()
