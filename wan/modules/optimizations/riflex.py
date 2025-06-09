import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from typing import Optional, Dict, Any, List, Tuple, Union
import math

from . import BaseConfig, OptimizationMetrics

__all__ = ['riflex_attention', 'RIFLEX_AVAILABLE']

RIFLEX_AVAILABLE = True  # Since this is our own implementation

class RIFLEXConfig(BaseConfig):
    """Configuration for RIFLEX optimization."""
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-5,
        use_rotary: bool = True,
        use_dynamic_ntk: bool = True,
        use_gated_mlp: bool = True,
        k: Optional[int] = None,
        L_test: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.use_rotary = use_rotary
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_gated_mlp = use_gated_mlp
        self.k = k  # For RIFLEX frequency adjustment
        self.L_test = L_test  # For RIFLEX frequency adjustment

class RotaryEmbedding(nn.Module):
    """Rotary position embeddings."""
    def __init__(self, dim: int, max_position_embeddings: int = 1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RIFLEXAttention(nn.Module):
    """RIFLEX attention module with rotary embeddings and dynamic NTK scaling."""
    def __init__(self, config: RIFLEXConfig):
        super().__init__()
        self.config = config
        
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
        if config.use_rotary:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=config.max_position_embeddings
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project queries, keys, values
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape and scale queries and keys
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if self.config.use_rotary:
            cos, sin = self.rotary_emb(q, seq_len=seq_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Apply dynamic NTK scaling
        if self.config.use_dynamic_ntk:
            scale = math.log(seq_len)
            attn_weights = attn_weights / scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        return attn_output

class RIFLEXMLP(nn.Module):
    """RIFLEX MLP module with optional gating."""
    def __init__(self, config: RIFLEXConfig):
        super().__init__()
        self.config = config
        
        self.fc1 = nn.Linear(config.hidden_dim, 4 * config.hidden_dim)
        if config.use_gated_mlp:
            self.gate = nn.Linear(config.hidden_dim, 4 * config.hidden_dim)
        self.fc2 = nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.activation_dropout)
        self.act = F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_gated_mlp:
            hidden = self.fc1(x)
            gate = torch.sigmoid(self.gate(x))
            hidden = hidden * gate
        else:
            hidden = self.fc1(x)
        
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        return hidden

class RIFLEXLayer(nn.Module):
    """RIFLEX transformer layer."""
    def __init__(self, config: RIFLEXConfig):
        super().__init__()
        self.attention = RIFLEXAttention(config)
        self.mlp = RIFLEXMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

def apply_riflex(
    model: nn.Module,
    config: RIFLEXConfig,
    target_modules: Optional[List[str]] = None
) -> nn.Module:
    """
    Apply RIFLEX optimization to a model.
    
    Args:
        model: Model to optimize
        config: RIFLEX configuration
        target_modules: List of module names to apply RIFLEX to
    
    Returns:
        Optimized model
    """
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.TransformerEncoderLayer):
                target_modules.append(name)
    
    # Create a new model to avoid modifying the original
    optimized_model = type(model)()
    optimized_model.load_state_dict(model.state_dict())
    
    # Replace target modules with RIFLEX layers
    for name in target_modules:
        parent_name = '.'.join(name.split('.')[:-1])
        module_name = name.split('.')[-1]
        parent = optimized_model if parent_name == '' else getattr(optimized_model, parent_name)
        setattr(parent, module_name, RIFLEXLayer(config))
    
    return optimized_model

def optimize_with_riflex(
    model: nn.Module,
    riflex_config: Optional[Dict[str, Any]] = None,
    hidden_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    activation_dropout: float = 0.1,
    max_position_embeddings: int = 1024,
    layer_norm_eps: float = 1e-5,
    use_rotary: bool = True,
    use_dynamic_ntk: bool = True,
    use_gated_mlp: bool = True,
    target_modules: Optional[List[str]] = None,
    k: Optional[int] = None,
    L_test: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply RIFLEX optimization to Wan2.1 model.
    
    Args:
        model: The model to optimize
        riflex_config: Optional RIFLEX configuration
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        activation_dropout: Activation dropout probability
        max_position_embeddings: Maximum sequence length
        layer_norm_eps: Layer normalization epsilon
        use_rotary: Whether to use rotary embeddings
        use_dynamic_ntk: Whether to use dynamic NTK scaling
        use_gated_mlp: Whether to use gated MLP
        target_modules: List of module names to apply RIFLEX to
        k: RIFLEX frequency adjustment parameter
        L_test: Number of frames for inference
    
    Returns:
        Tuple of (optimized_model, metrics)
    """
    # Initialize metrics tracking
    metrics = OptimizationMetrics()
    metrics.start()
    
    # Infer hidden_dim and num_heads if not provided
    if hidden_dim is None or num_heads is None:
        for module in model.modules():
            if hasattr(module, "hidden_size"):
                hidden_dim = hidden_dim or module.hidden_size
            if hasattr(module, "num_attention_heads"):
                num_heads = num_heads or module.num_attention_heads
    
    # Use default values if still not found
    hidden_dim = hidden_dim or 768
    num_heads = num_heads or 12
    
    # Create configuration
    config = RIFLEXConfig(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        max_position_embeddings=max_position_embeddings,
        layer_norm_eps=layer_norm_eps,
        use_rotary=use_rotary,
        use_dynamic_ntk=use_dynamic_ntk,
        use_gated_mlp=use_gated_mlp,
        k=k,
        L_test=L_test,
    ) if riflex_config is None else RIFLEXConfig.from_dict(riflex_config)
    
    # Apply optimization
    optimized_model = apply_riflex(
        model=model,
        config=config,
        target_modules=target_modules
    )
    
    # Stop metrics tracking and add config info
    metrics.stop()
    metrics.add_metric("config", config.to_dict())
    
    return optimized_model, metrics.get_metrics()

@amp.autocast(enabled=False)
def rope_params_riflex(
    max_seq_len: int,
    dim: int,
    theta: float = 10000,
    k: Optional[int] = None,
    L_test: Optional[int] = None
) -> torch.Tensor:
    """
    Modified RoPE parameters with RIFLEX's frequency adjustment.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embeddings
        theta: Base for the exponential
        k: Index for the intrinsic frequency in RoPE
        L_test: Number of frames for inference
    
    Returns:
        Complex tensor containing the RoPE frequencies
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    
    # RIFLEX modification: Reduce intrinsic frequency to stay within a single period
    if k is not None and L_test is not None:
        # Multiply by 0.9 to keep extrapolated length below 90% of a period
        freqs[:, k-1] = 0.9 * 2 * torch.pi / L_test
    
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs 