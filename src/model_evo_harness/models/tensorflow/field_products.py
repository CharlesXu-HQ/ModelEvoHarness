"""Independent TensorFlow field products; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Sources: model_zoo/{FFM,ONN,HOFM,HFM,DLRM,LorentzFM,DeepIM} and the source
interaction_machine primitive. Host owns embeddings, heads and training.
Pairs use i<j row-major order. Masks are bool True=valid, with NaN padding
removed before arithmetic. No external model-framework dependency is used.
"""

from numbers import Integral
import tensorflow as tf


def _dims(*dims):
    if any(not isinstance(d, Integral) or isinstance(d, bool) or d < 1 for d in dims):
        raise ValueError("dimensions must be positive integers")


def _tokens(x, mask, fields, dim):
    x = tf.ensure_shape(x, (None, fields, dim))
    if not x.dtype.is_floating:
        raise ValueError("tokens must be real floating tensors")
    if x.dtype in (tf.float16, tf.bfloat16):
        x = tf.cast(x, tf.float32)
    if mask is None:
        mask = tf.ones(tf.shape(x)[:2], tf.bool)
    if mask.dtype != tf.bool:
        raise ValueError("valid_mask must be bool [B,num_fields]")
    mask = tf.ensure_shape(mask, (None, fields))
    tf.debugging.assert_equal(tf.shape(mask)[0], tf.shape(x)[0])
    x = tf.where(mask[..., None], x, 0)
    return tf.debugging.check_numerics(x, "observed tokens must be finite"), mask


class FieldAwareFactorization(tf.keras.layers.Layer):
    """FFM partner-specific [B,F,F,D] -> pair dots [B,F(F-1)/2].

    The host owns partner-specific embedding parameters; repeating ordinary
    embeddings gives only an FM control. Diagonal entries are unused here.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2:
            raise ValueError("pair interactions require at least two fields")
        self.num_fields, self.dim = num_fields, dim
        self.pairs = [(i,j) for i in range(num_fields) for j in range(i+1,num_fields)]

    def _prepare(self, embeddings, valid_mask):
        embeddings = tf.ensure_shape(embeddings, (None, self.num_fields, self.num_fields, self.dim))
        if valid_mask is not None:
            if valid_mask.dtype != tf.bool:
                raise ValueError("valid_mask must be bool [B,num_fields]")
            valid_mask = tf.ensure_shape(valid_mask,(None,self.num_fields))
            tf.debugging.assert_equal(tf.shape(valid_mask)[0],tf.shape(embeddings)[0])
            embeddings = tf.where(valid_mask[:,None,:,None],embeddings,0)
        flat, mask = _tokens(tf.reshape(embeddings, (-1, self.num_fields, self.num_fields*self.dim)),
                             valid_mask, self.num_fields, self.num_fields*self.dim)
        x = tf.reshape(flat, tf.shape(embeddings))
        return tf.where(mask[:, None, :, None], x, 0), mask

    def call(self, embeddings, valid_mask=None):
        x, _ = self._prepare(embeddings, valid_mask)
        return tf.stack([tf.reduce_sum(x[:, i, j]*x[:, j, i], -1) for i,j in self.pairs], -1)


class OperationAwareInteraction(FieldAwareFactorization):
    """ONNv2: diagonal copy embeddings followed by partner-specific pair dots.

    [B,F,F,D] -> [B,F*D+P]. Distinct copy/pair operation embeddings must come
    from the host. No DNN, embedding table or prediction head is included.
    """
    def call(self, embeddings, valid_mask=None):
        x, _ = self._prepare(embeddings, valid_mask)
        diagonal = tf.reshape(tf.stack([x[:, i, i] for i in range(self.num_fields)], 1),
                              (-1, self.num_fields*self.dim))
        dots = tf.stack([tf.reduce_sum(x[:, i, j]*x[:, j, i], -1) for i,j in self.pairs], -1)
        return tf.concat((diagonal, dots), -1)


class HigherOrderFactorization(tf.keras.layers.Layer):
    """HOFM exact order-k distinct-field products via dynamic programming.

    [B,F,D] -> [B,D]. One supplied embedding bank and one order; the host can
    combine orders or independent banks. No linear term or task head.
    """
    def __init__(self, num_fields, dim, order):
        super().__init__()
        _dims(num_fields, dim, order)
        if order > num_fields:
            raise ValueError("order cannot exceed num_fields")
        self.num_fields, self.dim, self.order = num_fields, dim, order

    def call(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        terms = [tf.ones_like(x[:, 0])] + [tf.zeros_like(x[:, 0]) for _ in range(self.order)]
        for i in range(self.num_fields):
            terms = [terms[0]] + [terms[k]+x[:, i]*terms[k-1] for k in range(1,self.order+1)]
        return terms[-1]


class InteractionMachine(tf.keras.layers.Layer):
    """DeepIM Newton/power-sum recurrence, orders 1..K concatenated [B,KD].

    These are HOFM's elementary symmetric polynomials with an all-order readout.
    K<=5; large magnitudes may cause cancellation. No BN, DNN or task head.
    """
    def __init__(self, num_fields, dim, max_order):
        super().__init__()
        _dims(num_fields, dim, max_order)
        if max_order > min(5, num_fields):
            raise ValueError("max_order must be <= min(5,num_fields)")
        self.num_fields, self.dim, self.max_order = num_fields, dim, max_order

    def call(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        powers = [None] + [tf.reduce_sum(x**k, 1) for k in range(1,self.max_order+1)]
        terms = [tf.ones_like(x[:, 0])]
        for k in range(1,self.max_order+1):
            terms.append(sum((-1)**(j-1)*terms[k-j]*powers[j] for j in range(1,k+1))/k)
        return tf.concat(terms[1:], -1)


class HolographicInteraction(tf.keras.layers.Layer):
    """HFM circular convolution/correlation of pair coordinates -> [B,P,D].

    Correlation conjugates the left spectrum. Explicit inverse length supports
    odd D; half inputs are promoted. This is not a temporal convolution.
    """
    def __init__(self, num_fields, dim, mode="convolution"):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2 or mode not in ("convolution", "correlation"):
            raise ValueError("need at least two fields and convolution/correlation mode")
        self.num_fields, self.dim, self.mode = num_fields, dim, mode
        self.pairs = [(i,j) for i in range(num_fields) for j in range(i+1,num_fields)]

    def call(self, tokens, valid_mask=None):
        x, _ = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        spectrum = tf.signal.rfft(x)
        values = []
        for i,j in self.pairs:
            left = tf.math.conj(spectrum[:, i]) if self.mode == "correlation" else spectrum[:, i]
            values.append(tf.signal.irfft(left*spectrum[:, j], fft_length=[self.dim]))
        return tf.stack(values, 1)


class DenseSparseInteraction(tf.keras.layers.Layer):
    """DLRM: all pair dots of [sparse;dense token], then append dense vector.

    dense [B,D], sparse [B,F,D], optional sparse mask -> [B,(F+1)F/2+D].
    Host supplies the bottom projection, embedding tables and top prediction
    network. This is a composition reference, not a new dot-product operator.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        self.num_fields, self.dim = num_fields, dim

    def call(self, dense, sparse, valid_mask=None):
        sparse, _ = _tokens(sparse, valid_mask, self.num_fields, self.dim)
        dense, _ = _tokens(dense[:, None, :], None, 1, self.dim)
        tf.debugging.assert_equal(tf.shape(dense)[0], tf.shape(sparse)[0])
        x = tf.concat((sparse, dense), 1)
        pairs = [tf.reduce_sum(x[:, i]*x[:, j], -1) for i in range(self.num_fields+1)
                 for j in range(i+1,self.num_fields+1)]
        return tf.concat((tf.stack(pairs, -1), dense[:, 0]), -1)


class LorentzInteraction(tf.keras.layers.Layer):
    """LorentzFM triangle terms [B,F,D] -> [B,P], curvature fixed at one.

    ti=sqrt(1+||xi||²), pair=1+(1-dot(xi,xj)-ti-tj)/(ti*tj).
    Float64 intermediates guard float32 squares; float64 inputs still need a
    finite numerical range. Missing pairs are zero. No manifold optimizer.
    """
    def __init__(self, num_fields, dim):
        super().__init__()
        _dims(num_fields, dim)
        if num_fields < 2:
            raise ValueError("at least two fields are required")
        self.num_fields, self.dim = num_fields, dim

    def call(self, tokens, valid_mask=None):
        x, mask = _tokens(tokens, valid_mask, self.num_fields, self.dim)
        work = tf.cast(x, tf.float64)
        time = tf.sqrt(1+tf.reduce_sum(work**2, -1))
        values = []
        for i in range(self.num_fields):
            for j in range(i+1,self.num_fields):
                score = 1+(1-tf.reduce_sum(work[:, i]*work[:, j], -1)-time[:, i]-time[:, j])/(time[:, i]*time[:, j])
                values.append(tf.where(mask[:, i] & mask[:, j], score, 0))
        return tf.cast(tf.stack(values, -1), x.dtype)
