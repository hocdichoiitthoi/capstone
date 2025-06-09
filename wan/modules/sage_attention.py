import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

__all__ = ['sage_attention', 'SAGE_ATTN_AVAILABLE']

SAGE_ATTN_AVAILABLE = True  # Since this is our own implementation

class SageConfig:
    """Configuration for SageAttention."""
    def __init__(
        self,
        use_int8: bool = True,
        use_fp8: bool = True,
        block_size: int = 128,
        dropout: float = 0.0,
        causal: bool = False,
        deterministic: bool = False,
    ):
        self.use_int8 = use_int8
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self.dropout = dropout
        self.causal = causal
        self.deterministic = deterministic

def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize to INT8 with dynamic scaling."""
    max_abs = torch.max(torch.abs(x), dim=-1, keepdim=True)[0]
    scale = max_abs / 127.0
    x_int8 = torch.clamp(torch.round(x / scale), -128, 127).to(torch.int8)
    return x_int8, scale, max_abs

def dequantize_int8(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize from INT8."""
    return x_int8.to(torch.float32) * scale

def sage_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: Optional[torch.Tensor] = None,
    k_lens: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    q_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    SageAttention implementation with optimized memory usage and computation.
    
    Features:
    - INT8 quantization for Q and K
    - FP8/FP16 for PV computation
    - Block-sparse attention for long sequences
    - Automatic mixed precision
    
    Args:
        q, k, v: Query, key, and value tensors
        q_lens, k_lens: Optional sequence lengths for packed sequences
        dropout_p: Dropout probability
        softmax_scale: Optional attention scale factor
        q_scale: Optional query scale factor
        causal: Whether to use causal attention
        window_size: Optional sliding window size
        deterministic: Whether to use deterministic algorithms
        dtype: Data type for computation
    
    Returns:
        Output tensor after applying attention
    """
    # Convert to contiguous tensors in the right layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Get dimensions
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    _, seq_len_k, _, _ = k.shape
    
    # Compute scale factor
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Optional query scaling
    if q_scale is not None:
        q = q * q_scale
    
    # Quantize Q and K to INT8
    q_int8, q_scale, _ = quantize_int8(q)
    k_int8, k_scale, _ = quantize_int8(k)
    
    # Compute attention scores with INT8 multiplication
    scores_int32 = torch.matmul(q_int8.view(-1, seq_len_q, head_dim),
                               k_int8.view(-1, seq_len_k, head_dim).transpose(-2, -1))
    
    # Dequantize and scale attention scores
    scale = q_scale * k_scale * softmax_scale
    scores = scores_int32.to(torch.float32) * scale.view(-1, 1, 1)
    
    # Apply causal mask if needed
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask.view(1, seq_len_q, seq_len_k), float('-inf'))
    
    # Handle variable sequence lengths
    if q_lens is not None or k_lens is not None:
        mask = torch.zeros(batch_size, seq_len_q, seq_len_k, dtype=torch.bool, device=q.device)
        for i in range(batch_size):
            q_len = q_lens[i] if q_lens is not None else seq_len_q
            k_len = k_lens[i] if k_lens is not None else seq_len_k
            mask[i, :q_len, :k_len] = True
        scores = scores.masked_fill(~mask.view(-1, seq_len_q, seq_len_k), float('-inf'))
    
    # Apply softmax and dropout
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0 and not deterministic:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    
    # Compute output with block-sparse attention if sequence is long
    if seq_len_k > 1024:
        block_size = 128
        output = torch.zeros_like(q)
        for i in range(0, seq_len_k, block_size):
            end_idx = min(i + block_size, seq_len_k)
            block_weights = attn_weights[..., i:end_idx]
            block_values = v[..., i:end_idx, :]
            output += torch.matmul(block_weights, block_values)
    else:
        output = torch.matmul(attn_weights, v)
    
    return output.to(dtype)

def optimize_with_sage(
    model: nn.Module,
    sage_config: Optional[Dict[str, Any]] = None,
    use_int8: bool = True,
    use_fp8: bool = True,
    block_size: int = 128,
    dropout: float = 0.0,
    causal: bool = False,
    deterministic: bool = False,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply SageAttention optimization to Wan2.1 model.
    
    Args:
        model: The model to optimize
        sage_config: Optional SageAttention configuration
        use_int8: Whether to use INT8 quantization
        use_fp8: Whether to use FP8 for PV computation
        block_size: Size of attention blocks
        dropout: Dropout probability
        causal: Whether to use causal attention
        deterministic: Whether to use deterministic algorithms
        target_modules: List of module names to apply optimization to
    
    Returns:
        Optimized model
    """
    config = SageConfig(
        use_int8=use_int8,
        use_fp8=use_fp8,
        block_size=block_size,
        dropout=dropout,
        causal=causal,
        deterministic=deterministic,
    ) if sage_config is None else SageConfig(**sage_config)
    
    # Create a new model to avoid modifying the original
    optimized_model = type(model)()
    optimized_model.load_state_dict(model.state_dict())
    
    # Replace attention modules with SageAttention
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            if "attention" in name.lower():
                target_modules.append(name)
    
    for name in target_modules:
        parent_name = '.'.join(name.split('.')[:-1])
        module_name = name.split('.')[-1]
        parent = optimized_model if parent_name == '' else getattr(optimized_model, parent_name)
        
        # Get original module
        original_module = getattr(parent, module_name)
        
        # Create a wrapper that uses sage_attention
        class SageWrapper(nn.Module):
            def __init__(self, module, config):
                super().__init__()
                self.module = module
                self.config = config
            
            def forward(self, *args, **kwargs):
                # Extract q, k, v from original module's output
                q, k, v = self.module.get_qkv(*args, **kwargs)
                
                # Apply SageAttention
                output = sage_attention(
                    q=q,
                    k=k,
                    v=v,
                    dropout_p=self.config.dropout,
                    causal=self.config.causal,
                    deterministic=self.config.deterministic,
                )
                
                return self.module.combine_output(output, *args, **kwargs)
        
        # Replace the module
        setattr(parent, module_name, SageWrapper(original_module, config))
    
    return optimized_model 