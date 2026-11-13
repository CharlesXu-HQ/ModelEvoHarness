"""Independent native field mechanisms; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Host owns data semantics, embeddings, heads and training. Boolean masks use
True=valid; masked nonfinite padding is sanitized before arithmetic. Sources
and deliberate mechanism boundaries are documented on each class.
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


def _convolve(x,mask,conv,kernel_size,padding):
    output = conv(tf.pad(x[:,:,:,None],[[0,0],[padding,padding],[0,0],[0,0]]))
    valid = tf.pad(tf.cast(mask[:,:,None],tf.float32),[[0,0],[padding,padding],[0,0]])
    observed = tf.nn.max_pool1d(valid,kernel_size,strides=1,padding="VALID")>0
    return output,observed[:,:,:,None]


class OrderedKMaxConvolution(tf.keras.layers.Layer):
    """One CCPM wide convolution + order-preserving k-max + tanh.

    [B,F,D] -> [B,k*D*C], flattened position/coordinate/channel. Top-k per
    coordinate/channel restores source order, breaking ties by earlier index.
    Invalid fields are zero-filled; empty windows cannot win and missing top-k
    slots become zero. Host owns meaningful field order, stack and head.
    Source: model_zoo/CCPM/src/CCPM.py, CCPM_ConvLayer.
    """
    def __init__(self,num_fields,dim,channels,kernel_size,k):
        super().__init__()
        _dims(num_fields,dim,channels,kernel_size,k)
        if k > num_fields+kernel_size-1:
            raise ValueError("k cannot exceed the wide convolution length")
        self.num_fields,self.dim,self.channels = num_fields,dim,channels
        self.kernel_size,self.k = kernel_size,k
        self.conv = tf.keras.layers.Conv2D(channels,(kernel_size,1),padding="valid")

    def call(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        values,observed = _convolve(x,mask,self.conv,self.kernel_size,self.kernel_size-1)
        values = tf.transpose(tf.where(observed,values,tf.cast(-float("inf"),values.dtype)),(0,2,3,1))
        selected = tf.sort(tf.argsort(values,axis=-1,direction="DESCENDING",stable=True)[...,:self.k],axis=-1)
        pooled = tf.gather(values,selected,batch_dims=3,axis=3)
        pooled = tf.tanh(tf.where(tf.math.is_finite(pooled),pooled,0))
        return tf.reshape(tf.transpose(pooled,(0,3,1,2)),(-1,self.k*self.dim*self.channels))


class ConvolutionalFieldGenerator(tf.keras.layers.Layer):
    """FGCNN one conv/tanh/pool/recombine stage -> generated fields [B,G,D].

    Odd kernel preserves F; pooling right-pads incomplete windows. Missing
    fields are zero inputs, empty windows are excluded, all-absent output is
    zero. Recombination mixes position/coordinate/channel. No BN, multi-stage
    stack, original-field concatenation, embedding bank or task head.
    Source: model_zoo/FGCNN/src/FGCNN.py, FGCNN_Layer.
    """
    def __init__(self,num_fields,dim,channels,kernel_size,pool_size,generated_fields):
        super().__init__()
        _dims(num_fields,dim,channels,kernel_size,pool_size,generated_fields)
        if kernel_size%2 != 1:
            raise ValueError("kernel_size must be odd for length-preserving convolution")
        self.num_fields,self.dim,self.channels = num_fields,dim,channels
        self.kernel_size,self.pool_size,self.generated_fields = kernel_size,pool_size,generated_fields
        self.pooled_fields = (num_fields+pool_size-1)//pool_size
        self.conv = tf.keras.layers.Conv2D(channels,(kernel_size,1),padding="valid")
        self.recombine = tf.keras.layers.Dense(generated_fields*dim)

    def call(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        values,observed = _convolve(x,mask,self.conv,self.kernel_size,self.kernel_size//2)
        values = tf.where(observed,tf.tanh(values),tf.cast(-float("inf"),values.dtype))
        right = self.pooled_fields*self.pool_size-self.num_fields
        values = tf.pad(values,[[0,0],[0,right],[0,0],[0,0]],constant_values=tf.cast(-float("inf"),values.dtype))
        pooled = tf.nn.max_pool2d(values,(self.pool_size,1),strides=(self.pool_size,1),padding="VALID")
        pooled = tf.where(tf.math.is_finite(pooled),pooled,0)
        output = tf.tanh(self.recombine(tf.reshape(pooled,(-1,self.pooled_fields*self.dim*self.channels))))
        output = tf.reshape(output,(-1,self.generated_fields,self.dim))
        return tf.where(tf.reduce_any(mask,1)[:,None,None],output,0)
