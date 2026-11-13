"""Independent native field mechanisms; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Host owns data semantics, embeddings, heads and training. Boolean masks use
True=valid; masked nonfinite padding is sanitized before arithmetic. Sources
and deliberate mechanism boundaries are documented on each class.
"""

from numbers import Integral
import math
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


def _bounds(epsilon, max_log):
    if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(max_log) or not 0 < max_log <= 60:
        raise ValueError("epsilon must be positive finite; max_log must lie in (0,60]")


class LogarithmicInteraction(tf.keras.layers.Layer):
    """AFN exp(clamp(sum_f orders[f,o]*log(max(abs(x),epsilon)))) -> [B,O,D].

    Missing fields are log-zero neutral; all-absent samples return zero. Sign
    is discarded; exponent clipping saturates gradients. The host supplies
    normalization/head; upstream log/exp BN and ensemble DNN are omitted.
    Source: model_zoo/AFN/src/AFN.py, logarithmic_net.
    """
    def __init__(self,num_fields,dim,out_fields,epsilon=1e-5,max_log=20.):
        super().__init__()
        _dims(num_fields,dim,out_fields)
        _bounds(epsilon,max_log)
        self.num_fields,self.dim,self.out_fields = num_fields,dim,out_fields
        self.epsilon,self.max_log = epsilon,max_log
        self.orders = self.add_weight(name="orders",shape=(num_fields,out_fields),initializer="glorot_uniform")

    def call(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        logs = tf.where(mask[...,None],tf.math.log(tf.maximum(tf.abs(x),self.epsilon)),0)
        exponent = tf.einsum("bfd,fo->bod",logs,self.orders)
        output = tf.exp(tf.clip_by_value(exponent,-self.max_log,self.max_log))
        return tf.where(tf.reduce_any(mask,1)[:,None,None],output,0)


class EulerInteraction(tf.keras.layers.Layer):
    """EulerNet magnitude/phase orders + shared implicit ReLU projection.

    real/imaginary [B,F,D] -> two [B,O,D] tensors. Zero coordinates have zero
    phase and floored magnitude; exponent clipping saturates gradients.
    Masked fields are neutral for explicit products; all-absent output zero.
    No initial Euler embedding conversion, dropout, normalization or head.
    Source: model_zoo/EulerNet/src/EulerNet.py, EulerInteractionLayer.
    """
    def __init__(self,num_fields,dim,out_fields,epsilon=1e-8,max_log=20.):
        super().__init__()
        _dims(num_fields,dim,out_fields)
        _bounds(epsilon,max_log)
        self.num_fields,self.dim,self.out_fields = num_fields,dim,out_fields
        self.epsilon,self.max_log = epsilon,max_log
        self.orders = self.add_weight(name="orders",shape=(num_fields,out_fields),initializer="glorot_uniform")
        self.log_bias = self.add_weight(name="log_bias",shape=(out_fields,dim),initializer="zeros")
        self.phase_bias = self.add_weight(name="phase_bias",shape=(out_fields,dim),initializer="zeros")
        self.implicit = tf.keras.layers.Dense(out_fields*dim)

    def call(self,real,imaginary,valid_mask=None):
        r,mask = _tokens(real,valid_mask,self.num_fields,self.dim)
        p,_ = _tokens(imaginary,mask,self.num_fields,self.dim)
        tf.debugging.assert_equal(tf.shape(p),tf.shape(r))
        radius2 = r*r+p*p
        nonzero = radius2 > self.epsilon
        phase = tf.atan2(tf.where(nonzero,p,0),tf.where(nonzero,r,1))
        log_radius = tf.where(mask[...,None],.5*tf.math.log(tf.maximum(radius2,self.epsilon)),0)
        exponent = tf.einsum("bfd,fo->bod",log_radius,self.orders)+self.log_bias
        angle = tf.einsum("bfd,fo->bod",phase,self.orders)+self.phase_bias
        magnitude = tf.exp(tf.clip_by_value(exponent,-self.max_log,self.max_log))
        r_out = tf.reshape(tf.nn.relu(self.implicit(tf.reshape(r,(-1,self.num_fields*self.dim)))),(-1,self.out_fields,self.dim))+magnitude*tf.cos(angle)
        p_out = tf.reshape(tf.nn.relu(self.implicit(tf.reshape(p,(-1,self.num_fields*self.dim)))),(-1,self.out_fields,self.dim))+magnitude*tf.sin(angle)
        present = tf.reduce_any(mask,1)[:,None,None]
        return tf.where(present,r_out,0),tf.where(present,p_out,0)


class GeneralizedInteraction(tf.keras.layers.Layer):
    """AOANet outer products + learned field/subspace fusion and full matrices.

    initial [B,F,D], current [B,S,D], optional masks -> [B,O,D].
    fusion[o,d,e]=sum_s,f alpha[s,f,o]*initial[f,d]*current[s,e]; multiply
    elementwise W[o,d,e] then contract h[o,e]. No DNN/head.
    Source: model_zoo/AOANet/src/AOANet.py, GeneralizedInteraction.
    """
    def __init__(self,num_fields,input_subspaces,dim,output_subspaces):
        super().__init__()
        _dims(num_fields,input_subspaces,dim,output_subspaces)
        self.num_fields,self.input_subspaces,self.dim = num_fields,input_subspaces,dim
        self.alpha = self.add_weight(name="alpha",shape=(input_subspaces,num_fields,output_subspaces),initializer="ones")
        self.kernel = self.add_weight(name="kernel",shape=(output_subspaces,dim,dim),
                                     initializer=lambda shape,dtype=None: tf.tile(tf.eye(dim,dtype=dtype)[None],[output_subspaces,1,1]))
        self.h = self.add_weight(name="h",shape=(output_subspaces,dim),initializer="ones")

    def call(self,initial,current,field_mask=None,subspace_mask=None):
        x,_ = _tokens(initial,field_mask,self.num_fields,self.dim)
        y,_ = _tokens(current,subspace_mask,self.input_subspaces,self.dim)
        tf.debugging.assert_equal(tf.shape(x)[0],tf.shape(y)[0])
        fusion = tf.einsum("bfd,bse,sfo->bode",x,y,self.alpha)
        return tf.reduce_sum(fusion*self.kernel*self.h[:,None,:],-1)
