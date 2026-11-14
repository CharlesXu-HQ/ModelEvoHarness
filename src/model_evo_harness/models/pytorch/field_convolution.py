"""Independent native field mechanisms; FuxiCTR 183da1c944ba8930fcdcd54114837c081959d748.

Host owns data semantics, embeddings, heads and training. Boolean masks use
True=valid; masked nonfinite padding is sanitized before arithmetic. Sources
and deliberate mechanism boundaries are documented on each class.
"""

from numbers import Integral
import torch
from torch import nn
from torch.nn import functional as F


def _dims(*dims):
    if any(not isinstance(d, Integral) or isinstance(d, bool) or d < 1 for d in dims):
        raise ValueError("dimensions must be positive integers")


def _tokens(x, mask, fields, dim):
    if x.ndim != 3 or x.shape[1:] != (fields, dim) or not x.is_floating_point():
        raise ValueError("tokens must be floating [B,num_fields,dim]")
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if mask is None:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
        raise ValueError("valid_mask must be bool [B,num_fields]")
    x = torch.where(mask[..., None], x, 0)
    if not torch.isfinite(x).all():
        raise ValueError("observed tokens must be finite")
    return x, mask


def _convolve(x,mask,conv,kernel_size,padding):
    output = conv(F.pad(x[:,None,:,:],(0,0,padding,padding)))
    observed = F.max_pool1d(F.pad(mask[:,None,:].float(),(padding,padding)),kernel_size,stride=1)>0
    return output,observed[:,:,:,None]


class OrderedKMaxConvolution(nn.Module):
    """One CCPM wide field convolution + order-preserving k-max + tanh.

    [B,F,D] -> [B,k*D*C], flattened position/coordinate/channel order. Kernel
    [height,1] never convolves across embedding coordinates. Top-k is selected
    independently per channel/coordinate, ties prefer earlier positions, then
    positions are restored to their original order. Invalid fields are zero
    inputs; windows with no observed input are excluded and zero padded if
    fewer than k remain. Host must justify a stable field order. No full stack
    or task head. Source: model_zoo/CCPM/src/CCPM.py, CCPM_ConvLayer.
    """
    def __init__(self,num_fields,dim,channels,kernel_size,k):
        super().__init__()
        _dims(num_fields,dim,channels,kernel_size,k)
        if k > num_fields+kernel_size-1:
            raise ValueError("k cannot exceed the wide convolution length")
        self.num_fields,self.dim,self.channels = num_fields,dim,channels
        self.kernel_size,self.k = kernel_size,k
        self.conv = nn.Conv2d(1,channels,(kernel_size,1))

    def forward(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        values,observed = _convolve(x,mask,self.conv,self.kernel_size,self.kernel_size-1)
        values = values.masked_fill(~observed,-torch.inf).permute(0,1,3,2)
        selected = values.argsort(dim=-1,descending=True,stable=True)[...,:self.k].sort(-1).values
        pooled = values.gather(-1,selected)
        pooled = torch.where(torch.isfinite(pooled),pooled,0).tanh()
        return pooled.permute(0,3,2,1).reshape(-1,self.k*self.dim*self.channels)


class ConvolutionalFieldGenerator(nn.Module):
    """One FGCNN conv/tanh/pool stage followed by learned field recombination.

    [B,F,D] -> [B,G,D], newly generated fields only. Odd convolution kernel
    preserves F; non-overlapping max pools right-pad incomplete final windows.
    Dense recombination mixes flattened position/coordinate/channel features.
    Fully absent samples return zero; missing input fields are zero-filled,
    and empty receptive fields cannot win pooling. No BN, multi-stage stack,
    original-field concatenation, interaction head or embedding bank is added.
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
        self.conv = nn.Conv2d(1,channels,(kernel_size,1))
        self.recombine = nn.Linear(self.pooled_fields*dim*channels,generated_fields*dim)

    def forward(self,tokens,valid_mask=None):
        x,mask = _tokens(tokens,valid_mask,self.num_fields,self.dim)
        values,observed = _convolve(x,mask,self.conv,self.kernel_size,self.kernel_size//2)
        values = values.tanh().masked_fill(~observed,-torch.inf)
        right = self.pooled_fields*self.pool_size-self.num_fields
        values = F.pad(values,(0,0,0,right),value=-torch.inf)
        pooled = F.max_pool2d(values,(self.pool_size,1))
        pooled = torch.where(torch.isfinite(pooled),pooled,0).permute(0,2,3,1)
        output = self.recombine(pooled.reshape(-1,self.pooled_fields*self.dim*self.channels)).tanh()
        output = output.reshape(-1,self.generated_fields,self.dim)
        return torch.where(mask.any(1)[:,None,None],output,0)
