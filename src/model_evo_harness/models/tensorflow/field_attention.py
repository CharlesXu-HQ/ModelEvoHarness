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


def _weights(logits,mask):
    logits = tf.where(mask,logits,tf.cast(-float("inf"),logits.dtype))
    logits = tf.where(tf.reduce_any(mask,-1,keepdims=True),logits,0)
    return tf.nn.softmax(logits,-1)*tf.cast(mask,logits.dtype)


class HierarchicalInteractionAggregation(tf.keras.layers.Layer):
    """InterHAt recurrent attention aggregation on host-contextualized X1.

    u_k=pool(X_k); X_next=X_k+u_k*X1; then attention across u_k -> [B,D].
    Initial self-attention/FFN, dropout and head are host-owned. Empty inputs
    return zero. Source: InterHAt.py forward and AttentionalAggregation.
    """
    def __init__(self,num_fields,dim,order=2):
        super().__init__()
        _dims(num_fields,dim,order)
        self.num_fields,self.dim,self.order = num_fields,dim,order
        self.scorers = [tf.keras.Sequential([tf.keras.layers.Dense(dim,activation="relu"),
                                            tf.keras.layers.Dense(1,use_bias=False)]) for _ in range(order)]
        self.order_scorer = tf.keras.Sequential([tf.keras.layers.Dense(dim,activation="relu"),
                                                tf.keras.layers.Dense(1,use_bias=False)])

    def call(self,tokens,valid_mask=None):
        initial,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        current,summaries = initial,[]
        for score in self.scorers:
            weights = _weights(tf.squeeze(score(current),-1),mask)
            summary = tf.reduce_sum(weights[...,None]*current,1)
            summaries.append(summary)
            current = current+summary[:,None,:]*initial
        stacked = tf.stack(summaries,1)
        available = tf.tile(tf.reduce_any(mask,1)[:,None],[1,self.order])
        weights = _weights(tf.squeeze(self.order_scorer(stacked),-1),available)
        return tf.reduce_sum(weights[...,None]*stacked,1)


class FieldGraphInteraction(tf.keras.layers.Layer):
    """One FiGNN graph update with field-specific W_in/W_out and residual.

    g=masked softmax(leaky(a[xi;xj])), no self edges;
    output_i=xi+W_in[i] sum_j g_ij W_out[j]xj. Empty neighbors keep residual.
    [B,F,D] -> [B,F,D]; return_attention adds [B,F,F]. No GRU, extra initial
    residual, graph reuse or head. Source: FiGNN.py GraphLayer/construction.
    """
    def __init__(self,num_fields,dim):
        super().__init__()
        _dims(num_fields,dim)
        self.num_fields,self.dim = num_fields,dim
        self.scorer = tf.keras.layers.Dense(1,use_bias=False)
        self.input_kernel = self.add_weight(name="input_kernel",shape=(num_fields,dim,dim),initializer="glorot_uniform")
        self.output_kernel = self.add_weight(name="output_kernel",shape=(num_fields,dim,dim),initializer="glorot_uniform")

    def call(self,tokens,valid_mask=None,return_attention=False):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        left = tf.tile(x[:,:,None,:],[1,1,self.num_fields,1])
        right = tf.tile(x[:,None,:,:],[1,self.num_fields,1,1])
        logits = tf.nn.leaky_relu(tf.squeeze(self.scorer(tf.concat((left,right),-1)),-1),alpha=.01)
        pairs = mask[:,:,None] & mask[:,None,:] & ~tf.eye(self.num_fields,dtype=tf.bool)
        graph = _weights(logits,pairs)
        sent = tf.einsum("fij,bfj->bfi",self.output_kernel,x)
        received = tf.linalg.matmul(graph,sent)
        output = x+tf.einsum("fij,bfj->bfi",self.input_kernel,received)
        return (output,graph) if return_attention else output


class DisentangledFieldAttention(tf.keras.layers.Layer):
    """DESTINE valid-field-centered pair attention + independent unary keys.

    D divisible by H; output [B,F,D], optional weights [B,H,F,F]. Pair and
    unary softmax each normalize separately (combined row sum two). Proper
    [B,H,F,d] transpose corrects source's direct head view. No residual/head.
    Source: DESTINE.py DisentangledSelfAttention. Empty rows are zero.
    """
    def __init__(self,num_fields,dim,num_heads=1):
        super().__init__()
        _dims(num_fields,dim,num_heads)
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_fields,self.dim,self.num_heads = num_fields,dim,num_heads
        self.query,self.key,self.value = [tf.keras.layers.Dense(dim) for _ in range(3)]
        self.unary = tf.keras.layers.Dense(num_heads)

    def call(self,tokens,valid_mask=None,return_attention=False):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        width = self.dim//self.num_heads
        def heads(layer):
            return tf.transpose(tf.reshape(layer(x),(-1,self.num_fields,self.num_heads,width)),(0,2,1,3))
        q,k,v = heads(self.query),heads(self.key),heads(self.value)
        valid = tf.cast(mask[:,None,:,None],q.dtype)
        count = tf.maximum(tf.reduce_sum(valid,2,keepdims=True),1)
        q = q-tf.reduce_sum(q*valid,2,keepdims=True)/count
        k = k-tf.reduce_sum(k*valid,2,keepdims=True)/count
        pairs = mask[:,None,:,None] & mask[:,None,None,:]
        pair_weights = _weights(tf.linalg.matmul(q,k,transpose_b=True)/width**.5,pairs)
        unary_weights = _weights(tf.transpose(self.unary(x),(0,2,1)),mask[:,None,:])
        weights = pair_weights+unary_weights[:,:,None,:]*valid
        output = tf.reshape(tf.transpose(tf.linalg.matmul(weights,v),(0,2,1,3)),(-1,self.num_fields,self.dim))
        return (output,weights) if return_attention else output


class SelfAttentionInteraction(tf.keras.layers.Layer):
    """SAM raw pair vectors 2A/2E [B,F,F,D] or summed 3A/3E [B,F,D].

    2 uses dot(xi,xj) times learned pair vectors (A) or xi*xj (E). 3 uses
    dot(xi,Kxj), sums j and optionally adds Qxi. Self pairs included; no
    softmax, dropout or final aggregation. Source: SAM.py SAM2A/E, SAM3A/E.
    """
    def __init__(self,num_fields,dim,mode="2E",residual=True):
        super().__init__()
        _dims(num_fields,dim)
        if mode not in ("2A","2E","3A","3E"):
            raise ValueError("mode must be 2A, 2E, 3A or 3E")
        self.num_fields,self.dim,self.mode = num_fields,dim,mode
        self.pair_vectors = self.add_weight(name="pair_vectors",shape=(num_fields,num_fields,dim),initializer="ones") if mode.endswith("A") else None
        self.key = tf.keras.layers.Dense(dim,use_bias=False) if mode.startswith("3") else None
        self.residual = tf.keras.layers.Dense(dim,use_bias=False) if self.key is not None and residual else None

    def call(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        key = self.key(x) if self.key is not None else x
        dots = tf.linalg.matmul(x,key,transpose_b=True)
        vectors = self.pair_vectors if self.pair_vectors is not None else x[:,:,None,:]*x[:,None,:,:]
        output = dots[...,None]*vectors
        if self.key is not None:
            output = tf.reduce_sum(output,2)
            if self.residual is not None:
                output = output+self.residual(x)
        return output
