"""Independent TensorFlow/Keras references for multi-task, scenario and slate models."""

import math

import tensorflow as tf


def _expert(output_dim: int) -> tf.keras.layers.Layer:
    return tf.keras.layers.Dense(output_dim, activation="relu")


class SharedBottom(tf.keras.Model):
    """Dense features [B,D] -> one raw logit per observed task [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.bottom = _expert(hidden_dim)
        self.heads = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        shared = self.bottom(x)
        return tf.concat([head(shared) for head in self.heads], axis=1)


class ESMM(tf.keras.Model):
    """Impression features [B,D] -> click, post-click and joint probabilities [B]."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.click_tower = tf.keras.Sequential([_expert(hidden_dim), tf.keras.layers.Dense(1)])
        self.conversion_tower = tf.keras.Sequential([_expert(hidden_dim), tf.keras.layers.Dense(1)])

    def call(self, x: tf.Tensor) -> dict[str, tf.Tensor]:
        click = tf.sigmoid(tf.squeeze(self.click_tower(x), axis=-1))
        post_click = tf.sigmoid(tf.squeeze(self.conversion_tower(x), axis=-1))
        return {"click": click, "post_click": post_click, "joint": click * post_click}


class PLE(tf.keras.Model):
    """CGC extraction: shared and task experts, then raw task logits [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, shared_experts: int,
                 task_experts: int, expert_dim: int, *, levels: int = 2):
        super().__init__()
        if min(num_tasks, shared_experts, task_experts, levels) < 1:
            raise ValueError("PLE needs positive tasks, experts and levels")
        self.num_tasks = num_tasks
        self.shared_experts = [[_expert(expert_dim) for _ in range(shared_experts)]
                               for _ in range(levels)]
        self.task_experts = [[[_expert(expert_dim) for _ in range(task_experts)]
                              for _ in range(num_tasks)] for _ in range(levels)]
        self.task_gates = [[tf.keras.layers.Dense(task_experts + shared_experts)
                            for _ in range(num_tasks)] for _ in range(levels)]
        self.shared_gates = [tf.keras.layers.Dense(shared_experts + num_tasks * task_experts)
                             for _ in range(levels - 1)]
        self.heads = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    @staticmethod
    def _mix(values: tf.Tensor, gate: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(values * tf.nn.softmax(gate, axis=1)[:, :, None], axis=1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        shared = x
        tasks = [x] * self.num_tasks
        for level, (shared_group, task_groups, task_gates) in enumerate(
                zip(self.shared_experts, self.task_experts, self.task_gates)):
            shared_values = tf.stack([expert(shared) for expert in shared_group], axis=1)
            task_values = [[expert(tasks[t]) for expert in task_groups[t]]
                           for t in range(self.num_tasks)]
            next_tasks = []
            for t in range(self.num_tasks):
                values = tf.concat([tf.stack(task_values[t], axis=1), shared_values], axis=1)
                next_tasks.append(self._mix(values, task_gates[t](tasks[t])))
            if level < len(self.shared_gates):
                all_values = tf.concat([shared_values, *[tf.stack(v, axis=1)
                                                        for v in task_values]], axis=1)
                shared = self._mix(all_values, self.shared_gates[level](shared))
            tasks = next_tasks
        return tf.concat([head(value) for head, value in zip(self.heads, tasks)], axis=1)


class AITM(tf.keras.Model):
    """Attention transfer across ordered tasks; features [B,D] -> logits [B,T]."""

    def __init__(self, input_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        if num_tasks < 2:
            raise ValueError("AITM needs at least two ordered tasks")
        self.task_encoders = [_expert(hidden_dim) for _ in range(num_tasks)]
        self.transfer = [tf.keras.layers.Dense(hidden_dim)
                         for _ in range(num_tasks - 1)]
        self.queries = [tf.keras.layers.Dense(hidden_dim)
                        for _ in range(num_tasks - 1)]
        self.keys = [tf.keras.layers.Dense(hidden_dim)
                     for _ in range(num_tasks - 1)]
        self.values = [tf.keras.layers.Dense(hidden_dim)
                       for _ in range(num_tasks - 1)]
        self.hidden_dim = hidden_dim
        self.heads = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    def _attend(self, previous: tf.Tensor, current: tf.Tensor,
                step: int) -> tuple[tf.Tensor, tf.Tensor]:
        candidates = tf.stack([self.transfer[step](previous), current], axis=1)
        score = tf.reduce_sum(self.queries[step](candidates) *
                              self.keys[step](candidates), axis=-1) / math.sqrt(self.hidden_dim)
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(self.values[step](candidates) * weights[:, :, None], axis=1), weights

    def attention_weights(self, previous: tf.Tensor, current: tf.Tensor,
                          step: int) -> tf.Tensor:
        """Return the two candidate weights for an adjacent task transition."""
        return self._attend(previous, current, step)[1]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        previous = None
        outputs = []
        for t, (encoder, head) in enumerate(zip(self.task_encoders, self.heads)):
            current = encoder(x)
            if previous is not None:
                current, _ = self._attend(previous, current, t - 1)
            outputs.append(head(current))
            previous = current
        return tf.concat(outputs, axis=1)


def aitm_loss(logits: tf.Tensor, labels: tf.Tensor, *, alpha: float = 1.0) -> tf.Tensor:
    """Binary loss plus ordered funnel penalty for fully observed end-to-end labels."""
    probabilities = tf.sigmoid(logits)
    order_penalty = tf.reduce_mean(tf.nn.relu(probabilities[:, 1:] - probabilities[:, :-1]))
    binary_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=logits))
    return binary_loss + alpha * order_penalty


class M2M(tf.keras.Model):
    """Task/expert/scenario views; (features [B,D], scenario [B,S]) -> [B,T] logits."""

    def __init__(self, input_dim: int, scenario_dim: int, num_tasks: int,
                 num_experts: int, expert_dim: int, hidden_dim: int):
        super().__init__()
        self.num_tasks, self.num_experts = num_tasks, num_experts
        self.experts = [_expert(expert_dim) for _ in range(num_experts)]
        self.meta_gates = tf.keras.layers.Dense(num_tasks * num_experts)
        self.meta_towers = [tf.keras.Sequential([_expert(hidden_dim), tf.keras.layers.Dense(1)])
                            for _ in range(num_tasks)]

    def call(self, x: tf.Tensor, scenario: tf.Tensor) -> tf.Tensor:
        values = tf.stack([expert(x) for expert in self.experts], axis=1)
        gates = tf.nn.softmax(tf.reshape(self.meta_gates(scenario),
                                        (-1, self.num_tasks, self.num_experts)), axis=-1)
        outputs = []
        for t, tower in enumerate(self.meta_towers):
            mixture = tf.reduce_sum(values * gates[:, t, :, None], axis=1)
            outputs.append(tower(tf.concat([mixture, scenario], axis=-1)))
        return tf.concat(outputs, axis=1)


class APG(tf.keras.Model):
    """Scenario-generated low-rank center matrix; (features, scenario) -> [B] logits."""

    def __init__(self, input_dim: int, scenario_dim: int, rank: int, hidden_dim: int):
        super().__init__()
        self.rank = rank
        self.input_projection = tf.keras.layers.Dense(rank)
        self.generator = tf.keras.layers.Dense(rank * rank)
        self.output_projection = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.head = tf.keras.layers.Dense(1)

    def generated_matrix(self, scenario: tf.Tensor) -> tf.Tensor:
        return tf.reshape(self.generator(scenario), (-1, self.rank, self.rank))

    def call(self, x: tf.Tensor, scenario: tf.Tensor) -> tf.Tensor:
        center = tf.squeeze(tf.matmul(self.generated_matrix(scenario),
                                      self.input_projection(x)[:, :, None]), axis=-1)
        return tf.squeeze(self.head(self.output_projection(center)), axis=-1)


class HMoE(tf.keras.Model):
    """Domain-routed experts; cross-domain tower scores have stopped gradients."""

    def __init__(self, input_dim: int, num_domains: int, num_experts: int, expert_dim: int):
        super().__init__()
        self.num_domains = num_domains
        self.experts = [_expert(expert_dim) for _ in range(num_experts)]
        self.domain_gates = [tf.keras.layers.Dense(num_experts) for _ in range(num_domains)]
        self.domain_towers = [tf.keras.layers.Dense(1) for _ in range(num_domains)]
        self.domain_weights = tf.keras.layers.Dense(num_domains)
        self.domain_bias = tf.keras.layers.Embedding(num_domains, num_domains)

    def call(self, x: tf.Tensor, domain: tf.Tensor) -> tf.Tensor:
        experts = tf.stack([expert(x) for expert in self.experts], axis=1)
        scores = []
        for gate, tower in zip(self.domain_gates, self.domain_towers):
            mixture = tf.reduce_sum(
                experts * tf.nn.softmax(gate(x), axis=1)[:, :, None], axis=1)
            scores.append(tower(mixture))
        scores = tf.concat(scores, axis=1)
        own = tf.one_hot(tf.cast(domain, tf.int32), self.num_domains, dtype=scores.dtype)
        routed = own * scores + (1 - own) * tf.stop_gradient(scores)
        weights = tf.nn.softmax(self.domain_weights(x) + self.domain_bias(domain), axis=1)
        return tf.reduce_sum(weights * routed, axis=1)


class PEPNet(tf.keras.Model):
    """EPNet input gate and per-task PPNet hidden gates; outputs [B,T] logits."""

    def __init__(self, input_dim: int, context_dim: int, num_tasks: int, hidden_dim: int):
        super().__init__()
        self.embedding_gate = tf.keras.layers.Dense(input_dim, activation="sigmoid")
        self.bottom = _expert(hidden_dim)
        self.task_gates = [tf.keras.layers.Dense(hidden_dim, activation="sigmoid")
                           for _ in range(num_tasks)]
        self.heads = [tf.keras.layers.Dense(1) for _ in range(num_tasks)]

    def call(self, x: tf.Tensor, ep_context: tf.Tensor,
             pp_context: tf.Tensor) -> tf.Tensor:
        personalized = x * (2 * self.embedding_gate(ep_context))
        hidden = self.bottom(personalized)
        return tf.concat([head(hidden * (2 * gate(pp_context)))
                          for gate, head in zip(self.task_gates, self.heads)], axis=1)


class PartitionedNorm(tf.keras.layers.Layer):
    """Normalize each domain over its batch in training and running moments in eval."""

    def __init__(self, input_dim: int, num_domains: int, *, momentum: float = 0.9):
        super().__init__()
        self.num_domains = num_domains
        self.momentum = momentum
        self.eps = 1e-5
        self.global_scale = self.add_weight(name="global_scale", shape=(input_dim,),
                                            initializer="ones")
        self.global_shift = self.add_weight(name="global_shift", shape=(input_dim,),
                                            initializer="zeros")
        self.domain_scale = tf.keras.layers.Embedding(
            num_domains, input_dim, embeddings_initializer="ones")
        self.domain_shift = tf.keras.layers.Embedding(
            num_domains, input_dim, embeddings_initializer="zeros")
        shape = (num_domains, input_dim)
        self.running_mean = self.add_weight(name="running_mean", shape=shape,
                                            initializer="zeros", trainable=False)
        self.running_var = self.add_weight(name="running_var", shape=shape,
                                           initializer="ones", trainable=False)
        self.running_batches = self.add_weight(name="running_batches", shape=(num_domains,),
                                               initializer="zeros", dtype="int32", trainable=False)

    def call(self, x: tf.Tensor, domain: tf.Tensor, training: bool | None = None) -> tf.Tensor:
        domain = tf.cast(domain, tf.int32)
        count = tf.math.unsorted_segment_sum(tf.ones_like(domain, dtype=x.dtype),
                                              domain, self.num_domains)
        sum_x = tf.math.unsorted_segment_sum(x, domain, self.num_domains)
        mean = sum_x / tf.maximum(count[:, None], 1)
        squared = tf.square(x - tf.gather(mean, domain))
        var = tf.math.unsorted_segment_sum(squared, domain, self.num_domains) / \
            tf.maximum(count[:, None], 1)
        valid = count >= 2
        seen = self.running_batches > 0

        def train_stats():
            next_mean = tf.where(seen[:, None],
                                 self.momentum * self.running_mean + (1 - self.momentum) * mean,
                                 mean)
            next_var = tf.where(seen[:, None],
                                self.momentum * self.running_var + (1 - self.momentum) * var,
                                var)
            self.running_mean.assign(tf.where(valid[:, None], next_mean, self.running_mean))
            self.running_var.assign(tf.where(valid[:, None], next_var, self.running_var))
            self.running_batches.assign_add(tf.cast(valid, tf.int32))
            return (tf.where(valid[:, None], mean, self.running_mean),
                    tf.where(valid[:, None], var, self.running_var), tf.logical_or(valid, seen))

        def eval_stats():
            return self.running_mean, self.running_var, seen

        mean, var, use_moments = tf.cond(tf.cast(False if training is None else training,
                                                tf.bool), train_stats, eval_stats)
        standardized = (x - tf.gather(mean, domain)) / \
            tf.sqrt(tf.gather(var, domain) + self.eps)
        normalized = tf.where(tf.gather(use_moments, domain)[:, None], standardized, x)
        return (normalized * self.global_scale * self.domain_scale(domain) +
                self.global_shift + self.domain_shift(domain))


class STAR(tf.keras.Model):
    """Partitioned normalization and shared × domain star layer; [B,D], [B] -> [B]."""

    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int):
        super().__init__()
        self.norm = PartitionedNorm(input_dim, num_domains)
        self.shared_weight = self.add_weight(name="shared_weight", shape=(input_dim, hidden_dim),
                                             initializer="glorot_uniform")
        self.domain_weight = self.add_weight(name="domain_weight",
                                             shape=(num_domains, input_dim, hidden_dim),
                                             initializer="ones")
        self.star_head = tf.keras.layers.Dense(1)
        self.aux_head = tf.keras.layers.Dense(1)
        self.aux_domain = tf.keras.layers.Embedding(num_domains, 1)

    def call(self, x: tf.Tensor, domain: tf.Tensor,
             training: bool | None = None) -> tf.Tensor:
        normalized = self.norm(x, domain, training=training)
        weights = self.shared_weight[None, :, :] * tf.gather(self.domain_weight, domain)
        hidden = tf.nn.relu(tf.einsum("bi,bij->bj", normalized, weights))
        return tf.squeeze(self.star_head(hidden) + self.aux_head(normalized) +
                          self.aux_domain(domain), axis=-1)


class PRM(tf.keras.Model):
    """Slate candidates [B,S,D], user [B,U], mask [B,S] -> position softmax [B,S]."""

    def __init__(self, input_dim: int, user_dim: int, hidden_dim: int,
                 max_positions: int, num_heads: int):
        super().__init__()
        self.item = tf.keras.layers.Dense(hidden_dim)
        self.user = tf.keras.layers.Dense(hidden_dim)
        self.position = tf.keras.layers.Embedding(max_positions, hidden_dim)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads)
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(1)

    def call(self, candidates: tf.Tensor, user: tf.Tensor,
             mask: tf.Tensor) -> tf.Tensor:
        assertion = tf.debugging.assert_positive(
            tf.reduce_min(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)),
            message="PRM needs at least one eligible candidate per slate")
        with tf.control_dependencies([assertion] if assertion is not None else []):
            mask = tf.identity(mask)
        position = self.position(tf.range(tf.shape(candidates)[1]))
        sequence = self.item(candidates) + self.user(user)[:, None, :] + position[None, :, :]
        allowed = tf.logical_and(mask[:, :, None], mask[:, None, :])
        attended = self.attention(sequence, sequence, attention_mask=allowed)
        logits = tf.squeeze(self.head(self.norm(sequence + attended)), axis=-1)
        return tf.nn.softmax(tf.where(mask, logits, tf.fill(tf.shape(logits),
                                                            tf.constant(-1e9, logits.dtype))), axis=1)


class PRS(tf.keras.Model):
    """Slate candidates [B,S,D] -> per-position CTR/continuation probabilities [B,S]."""

    def __init__(self, input_dim: int, hidden_dim: int, max_positions: int):
        super().__init__()
        self.item = tf.keras.layers.Dense(hidden_dim)
        self.position = tf.keras.layers.Embedding(max_positions, hidden_dim)
        self.ctr_head = tf.keras.layers.Dense(1, activation="sigmoid")
        self.continuation_head = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, candidates: tf.Tensor) -> dict[str, tf.Tensor]:
        position = self.position(tf.range(tf.shape(candidates)[1]))
        hidden = tf.nn.relu(self.item(candidates) + position[None, :, :])
        return {"ctr": tf.squeeze(self.ctr_head(hidden), axis=-1),
                "continuation": tf.squeeze(self.continuation_head(hidden), axis=-1)}

    @staticmethod
    def slate_reward(ctr: tf.Tensor, continuation: tf.Tensor) -> tf.Tensor:
        reach = tf.math.cumprod(tf.concat([tf.ones_like(continuation[:, :1]),
                                           continuation[:, :-1]], axis=1), axis=1)
        return tf.reduce_sum(ctr * reach, axis=1)

    def rerank(self, candidates: tf.Tensor, *, beam_size: int = 3) -> tf.Tensor:
        """Beam-propose permutations, then rescore complete slates; eager inference only."""
        if beam_size < 1:
            raise ValueError("beam_size must be positive")
        orders = []
        for slate in candidates:
            beams = [((), 0.0)]
            for _ in range(len(slate)):
                expanded = []
                for prefix, _ in beams:
                    for index in range(len(slate)):
                        if index in prefix:
                            continue
                        order = (*prefix, index)
                        output = self(tf.gather(slate, order)[None, :, :], training=False)
                        value = float(self.slate_reward(output["ctr"], output["continuation"])[0])
                        expanded.append((order, value))
                beams = sorted(expanded, key=lambda pair: (-pair[1], pair[0]))[:beam_size]
            rescored = []
            for order, _ in beams:
                output = self(tf.gather(slate, order)[None, :, :], training=False)
                value = float(self.slate_reward(output["ctr"], output["continuation"])[0])
                rescored.append((order, value))
            orders.append(max(rescored, key=lambda pair: pair[1])[0])
        return tf.constant(orders, dtype=tf.int32)
