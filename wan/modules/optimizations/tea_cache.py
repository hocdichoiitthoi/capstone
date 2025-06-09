import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict
import math

from . import BaseConfig, OptimizationMetrics

__all__ = ['optimize_with_teacache', 'TEACACHE_AVAILABLE']

TEACACHE_AVAILABLE = True  # Since this is our own implementation

class TeaCacheConfig(BaseConfig):
    """Configuration for TeaCache optimization."""
    def __init__(
        self,
        cache_size: int = 8192,
        cache_type: str = "lru",
        score_threshold: float = 0.5,
        use_fp16: bool = False,
        prune_rate: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.cache_size = cache_size
        self.cache_type = cache_type
        self.score_threshold = score_threshold
        self.use_fp16 = use_fp16
        self.prune_rate = prune_rate
        self.device = device

class TeaCache:
    """TeaCache implementation for caching attention computations."""
    def __init__(self, config: TeaCacheConfig):
        self.config = config
        self.cache = OrderedDict()
        self.scores = {}  # For score-based replacement
    
    def _compute_key(self, query: torch.Tensor, key: torch.Tensor) -> str:
        """Compute cache key from query and key tensors."""
        return f"{hash(query.data.cpu().numpy().tobytes())}_{hash(key.data.cpu().numpy().tobytes())}"
    
    def _update_score(self, key: str, attn_weights: torch.Tensor) -> None:
        """Update importance score for cached item."""
        if self.config.cache_type == "score":
            score = torch.mean(torch.abs(attn_weights)).item()
            self.scores[key] = score
    
    def _prune_cache(self) -> None:
        """Remove least important items from cache."""
        if len(self.cache) <= self.config.cache_size:
            return
            
        num_to_remove = int(len(self.cache) * self.config.prune_rate)
        
        if self.config.cache_type == "lru":
            # Remove oldest entries
            for _ in range(num_to_remove):
                self.cache.popitem(last=False)
        else:  # score-based
            # Remove entries with lowest scores
            sorted_keys = sorted(self.scores.items(), key=lambda x: x[1])
            for key, _ in sorted_keys[:num_to_remove]:
                self.cache.pop(key, None)
                self.scores.pop(key, None)
    
    def get(self, query: torch.Tensor, key: torch.Tensor) -> Optional[torch.Tensor]:
        """Get cached attention output if available."""
        cache_key = self._compute_key(query, key)
        if cache_key in self.cache:
            # Move to end for LRU
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        return None
    
    def put(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_weights: torch.Tensor) -> None:
        """Cache attention computation result."""
        cache_key = self._compute_key(query, key)
        
        # Convert to FP16 if enabled
        if self.config.use_fp16:
            value = value.half()
        
        self.cache[cache_key] = value
        self._update_score(cache_key, attn_weights)
        self._prune_cache()

class TeaCacheWrapper(nn.Module):
    """Wrapper for TeaCache optimization."""
    def __init__(self, module: nn.Module, config: TeaCacheConfig):
        super().__init__()
        self.module = module
        self.config = config
        self.cache = TeaCache(config)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with TeaCache optimization."""
        # Check cache first
        cached_output = self.cache.get(query, key)
        if cached_output is not None:
            return cached_output
        
        # Compute attention if not cached
        output, attn_weights = self.module(query, key, value, *args, **kwargs)
        
        # Cache result
        self.cache.put(query, key, output, attn_weights)
        
        return output

def optimize_with_teacache(
    model: nn.Module,
    teacache_config: Optional[Dict[str, Any]] = None,
    cache_size: int = 8192,
    cache_type: str = "lru",
    score_threshold: float = 0.5,
    use_fp16: bool = False,
    prune_rate: float = 0.1,
    device: str = "cuda",
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply TeaCache optimization to Wan2.1 model.
    
    Args:
        model: The model to optimize
        teacache_config: Optional TeaCache configuration
        cache_size: Size of attention cache
        cache_type: Cache replacement strategy ("lru" or "score")
        score_threshold: Threshold for score-based caching
        use_fp16: Whether to use FP16 for cached values
        prune_rate: Rate at which to prune cache
        device: Device to store cache on
        target_modules: List of module names to apply optimization to
    
    Returns:
        Tuple of (optimized_model, metrics)
    """
    # Initialize metrics tracking
    metrics = OptimizationMetrics()
    metrics.start()
    
    # Create configuration
    config = TeaCacheConfig(
        cache_size=cache_size,
        cache_type=cache_type,
        score_threshold=score_threshold,
        use_fp16=use_fp16,
        prune_rate=prune_rate,
        device=device,
    ) if teacache_config is None else TeaCacheConfig.from_dict(teacache_config)
    
    # Create a new model to avoid modifying the original
    optimized_model = type(model)()
    optimized_model.load_state_dict(model.state_dict())
    
    # Replace target modules with TeaCache wrapper
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
        
        # Replace with TeaCache wrapper
        setattr(parent, module_name, TeaCacheWrapper(original_module, config))
    
    # Stop metrics tracking and add config info
    metrics.stop()
    metrics.add_metric("config", config.to_dict())
    metrics.add_metric("num_optimized_modules", len(target_modules))
    
    return optimized_model, metrics.get_metrics() 