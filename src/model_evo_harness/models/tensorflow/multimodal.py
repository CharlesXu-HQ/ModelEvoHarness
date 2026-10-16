"""Paper-derived recommendation heads over supplied representations, without encoders."""

import tensorflow as tf


def _attention(logits, mask):
    active = tf.cast(mask, tf.bool)
    work = tf.cast(logits, tf.float32) if logits.dtype in (tf.float16, tf.bfloat16) else logits
    masked = tf.where(active, work, tf.cast(float("-inf"), work.dtype))
    masked = tf.where(tf.reduce_any(active, -1, keepdims=True), masked, tf.zeros_like(masked))
    return tf.cast(tf.where(active, tf.nn.softmax(masked, -1), tf.zeros_like(masked)), logits.dtype)


class DICM(tf.keras.layers.Layer):
    """Image Matters two-query attentive pooling [B,T,D] -> [B,2D].

    Candidate image [B,D] and ID [B,Q] query the same history with separate MLPs.
    Hidden width/activation are a compact local choice, not a fully specified
    paper configuration. Host owns encoders, CTR head and AMS; this head depends
    on the candidate and cannot be precomputed as an independent retrieval tower.
    """

    def __init__(self, image_dim: int, id_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.image_attention = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation="tanh"),
                                                    tf.keras.layers.Dense(1)])
        self.id_attention = tf.keras.Sequential([tf.keras.layers.Dense(hidden_dim, activation="tanh"),
                                                 tf.keras.layers.Dense(1)])
        self.image_attention.build((None, None, 2 * image_dim))
        self.id_attention.build((None, None, image_dim + id_dim))

    def call(self, history, image_query, id_query, history_mask):
        outputs = []
        for query, scorer in ((image_query, self.image_attention), (id_query, self.id_attention)):
            repeated = tf.repeat(query[:, None, :], tf.shape(history)[1], axis=1)
            logits = tf.squeeze(scorer(tf.concat((history, repeated), -1)), -1)
            outputs.append(tf.reduce_sum(_attention(logits, history_mask)[..., None] * history, 1))
        return tf.concat(outputs, -1)


class MMRec(tf.keras.layers.Layer):
    """Four cross-modal attention streams, independently normalized and then summed.

    Text/image histories [B,T,D], candidate text/image [B,D], mask [B,T]
    -> score [B]. Encoders, ROI detection and ViLBERT remain outside the head.
    """

    def representation(self, history_text, history_image, candidate_text, candidate_image, history_mask):
        user = tf.zeros_like(candidate_text)
        for history in (history_text, history_image):
            for candidate in (candidate_text, candidate_image):
                logits = tf.reduce_sum(history * candidate[:, None, :], -1)
                user += tf.reduce_sum(_attention(logits, history_mask)[..., None] * history, 1)
        return user

    def call(self, history_text, history_image, candidate_text, candidate_image, history_mask):
        user = self.representation(history_text, history_image, candidate_text, candidate_image, history_mask)
        return tf.reduce_sum(user * (candidate_text + candidate_image), -1)


class DMTLDistillation(tf.keras.layers.Layer):
    """Mean Bernoulli KL from detached product teacher; [B] logits -> scalar.

    Teacher is p(click)*p(long read|click); student is a supplied two-tower logit.
    Host owns MMoE teacher, student, duration label definition and training schedule.
    """

    def call(self, click_logits, conditional_read_logits, student_logits):
        teacher = tf.stop_gradient(tf.sigmoid(click_logits) * tf.sigmoid(conditional_read_logits))
        entropy = -tf.math.xlogy(teacher, teacher) - tf.math.xlogy(1 - teacher, 1 - teacher)
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=teacher, logits=student_logits)
        return tf.reduce_mean(cross_entropy - entropy)

    def teacher_loss(self, click_logits, conditional_read_logits, click_labels, joint_read_labels):
        """Two mean BCEs; joint-read is an impression label, not conditional CVR.

        Use click labels for Eq. 8 according to the prose/Figure 1, resolving
        the apparent swapped-label typo in the displayed equation.
        """
        a, b = click_logits, conditional_read_logits
        joint_logits = a + b - tf.reduce_logsumexp(tf.stack((tf.zeros_like(a), a, b)), 0)
        return (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=click_labels, logits=a))
                + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=joint_read_labels, logits=joint_logits)))
