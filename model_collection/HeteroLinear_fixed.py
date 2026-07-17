"""
Fixed version of torch_geometric.nn.dense.HeteroLinear that properly handles
multi-GPU device switching by clearing timing cache and using device-specific
CUDA synchronization.

Based on PyTorch Geometric's linear.py but with critical fixes for CUDA
illegal memory access errors when switching between devices.
"""
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter

import torch_geometric.backend
import torch_geometric.typing
from torch_geometric import is_compiling
from torch_geometric.index import index2ptr
from torch_geometric.nn import inits
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import index_sort


# Disable timing cache via environment variable
os.environ['PYG_DISABLE_TIMING_CACHE'] = '1'


def is_uninitialized_parameter(x) -> bool:
    if not hasattr(torch.nn.parameter, 'UninitializedParameter'):
        return False
    return isinstance(x, torch.nn.parameter.UninitializedParameter)


def reset_weight_(weight: Tensor, in_channels: int,
                  initializer: Optional[str] = None) -> Tensor:
    if in_channels <= 0:
        pass
    elif initializer == 'glorot':
        inits.glorot(weight)
    elif initializer == 'uniform':
        bound = 1.0 / math.sqrt(in_channels)
        torch.nn.init.uniform_(weight.data, -bound, bound)
    elif initializer == 'kaiming_uniform':
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    elif initializer is None:
        inits.kaiming_uniform(weight, fan=in_channels, a=math.sqrt(5))
    else:
        raise RuntimeError(f"Weight initializer '{initializer}' not supported")
    return weight


def reset_bias_(bias: Optional[Tensor], in_channels: int,
                initializer: Optional[str] = None) -> Optional[Tensor]:
    if bias is None or in_channels <= 0:
        pass
    elif initializer == 'zeros':
        inits.zeros(bias)
    elif initializer is None:
        inits.uniform(in_channels, bias)
    else:
        raise RuntimeError(f"Bias initializer '{initializer}' not supported")
    return bias


class HeteroLinear(torch.nn.Module):
    r"""Applies separate linear transformations to the incoming data according
    to types.

    **FIXED VERSION**: This version properly handles device switching in multi-GPU
    setups by clearing timing cache when device changes and using device-specific
    CUDA synchronization.

    For type :math:`\kappa`, it computes

    .. math::
        \mathbf{x}^{\prime}_{\kappa} = \mathbf{x}_{\kappa}
        \mathbf{W}^{\top}_{\kappa} + \mathbf{b}_{\kappa}.

    It supports lazy initialization and customizable weight and bias
    initialization.

    Args:
        in_channels (int): Size of each input sample. Will be initialized
            lazily in case it is given as :obj:`-1`.
        out_channels (int): Size of each output sample.
        num_types (int): The number of types.
        is_sorted (bool, optional): If set to :obj:`True`, assumes that
            :obj:`type_vec` is sorted. This avoids internal re-sorting of the
            data and can improve runtime and memory efficiency.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments.

    Shapes:
        - **input:**
          features :math:`(*, F_{in})`,
          type vector :math:`(*)`
        - **output:** features :math:`(*, F_{out})`
    """
    _timing_cache: Dict[int, Tuple[float, float]]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_types: int,
        is_sorted: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_types = num_types
        self.is_sorted = is_sorted
        self.kwargs = kwargs

        # Track the last device this module was used on
        self._last_device = None

        if self.in_channels == -1:
            self.weight = torch.nn.parameter.UninitializedParameter()
            self._hook = self.register_forward_pre_hook(
                self.initialize_parameters)
        else:
            self.weight = torch.nn.Parameter(
                torch.empty(num_types, in_channels, out_channels))

        if kwargs.get('bias', True):
            self.bias = Parameter(torch.empty(num_types, out_channels))
        else:
            self.register_parameter('bias', None)

        # Timing cache for benchmarking naive vs. segment matmul usage
        self._timing_cache: Dict[int, Tuple[float, float]] = {}

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset_weight_(self.weight, self.in_channels,
                      self.kwargs.get('weight_initializer', None))
        reset_bias_(self.bias, self.in_channels,
                    self.kwargs.get('bias_initializer', None))

    def forward_naive(self, x: Tensor, type_ptr: Tensor) -> Tensor:
        out = x.new_empty(x.size(0), self.out_channels)
        for i, (start, end) in enumerate(zip(type_ptr[:-1], type_ptr[1:])):
            out[start:end] = x[start:end] @ self.weight[i]
        return out

    def forward_segmm(self, x: Tensor, type_ptr: Tensor) -> Tensor:
        return pyg_lib.ops.segment_matmul(x, type_ptr, self.weight)

    @torch.no_grad()
    def _update_timing_cache(
        self,
        x: Tensor,
        type_ptr: Tensor,
        key: int,
    ) -> None:
        """
        **FIXED**: Use device-specific CUDA synchronization to prevent illegal
        memory access errors when switching between GPUs.
        """
        MEASURE_ITER = 1 if 'pytest' in sys.modules else 3
        
        # Get the device from input tensor
        device = x.device

        # Device-specific synchronization
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        for _ in range(MEASURE_ITER):
            _ = self.forward_segmm(x, type_ptr)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        time_segmm = time.perf_counter() - t

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t = time.perf_counter()
        for _ in range(MEASURE_ITER):
            _ = self.forward_naive(x, type_ptr)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        time_naive = time.perf_counter() - t

        self._timing_cache[key] = (time_segmm, time_naive)

    def forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        r"""The forward pass.

        Args:
            x (torch.Tensor): The input features.
            type_vec (torch.Tensor): A vector that maps each entry to a type.
        """
        # **CRITICAL FIX**: Clear timing cache if device changed
        current_device = x.device
        if self._last_device is not None and self._last_device != current_device:
            self._timing_cache.clear()
        self._last_device = current_device
        
        perm: Optional[Tensor] = None
        if not self.is_sorted and (type_vec[1:] < type_vec[:-1]).any():
            type_vec, perm = index_sort(type_vec, self.num_types)
            x = x[perm]

        type_ptr = index2ptr(type_vec, self.num_types)

        # Disable timing-based selection when environment variable is set
        if os.environ.get('PYG_DISABLE_TIMING_CACHE') == '1':
            use_segment_matmul = False
        elif torch_geometric.backend.use_segment_matmul is None:
            use_segment_matmul = False
            if (torch_geometric.typing.WITH_SEGMM and not is_compiling()
                    and not torch.jit.is_scripting()):

                # Use "magnitude" of number of rows as timing key
                key = math.floor(math.log10(x.size(0)))
                if key not in self._timing_cache:
                    self._update_timing_cache(x, type_ptr, key)
                time_segmm, time_naive = self._timing_cache[key]
                use_segment_matmul = time_segmm < time_naive
        else:
            use_segment_matmul = torch_geometric.backend.use_segment_matmul

        if (torch_geometric.typing.WITH_SEGMM and not is_compiling()
                and use_segment_matmul):
            out = self.forward_segmm(x, type_ptr)
        else:
            out = self.forward_naive(x, type_ptr)

        if self.bias is not None:
            out += self.bias[type_vec]

        if perm is not None:  # Restore original order (if necessary)
            out_unsorted = torch.empty_like(out)
            out_unsorted[perm] = out
            out = out_unsorted

        return out

    @torch.no_grad()
    def initialize_parameters(self, module, input):
        if is_uninitialized_parameter(self.weight):
            self.in_channels = input[0].size(-1)
            self.weight.materialize(
                (self.num_types, self.in_channels, self.out_channels))
            self.reset_parameters()
        self._hook.remove()
        delattr(self, '_hook')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_types={self.num_types}, '
                f'bias={self.kwargs.get("bias", True)})')
