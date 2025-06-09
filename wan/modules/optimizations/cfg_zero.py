import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import math

from . import BaseConfig, OptimizationMetrics

__all__ = ['optimize_with_cfg_zero', 'CFG_ZERO_AVAILABLE']

CFG_ZERO_AVAILABLE = True  # Since this is our own implementation

class CFGZeroConfig(BaseConfig):
    """Configuration for CFG-Zero optimization."""
    def __init__(
        self,
        mode: str = "dynamic",
        optimize_memory: bool = True,
        guidance_scale: float = 7.5,
        min_guidance: float = 1.0,
        max_guidance: float = 20.0,
        dynamic_threshold: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.optimize_memory = optimize_memory
        self.guidance_scale = guidance_scale
        self.min_guidance = min_guidance
        self.max_guidance = max_guidance
        self.dynamic_threshold = dynamic_threshold

class CFGZeroWrapper(nn.Module):
    """Wrapper for CFG-Zero optimization."""
    def __init__(self, module: nn.Module, config: CFGZeroConfig):
        super().__init__()
        self.module = module
        self.config = config
        self.cached_cond = None
        self.cached_uncond = None
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, uncond: torch.Tensor) -> torch.Tensor:
        """Forward pass with CFG-Zero optimization."""
        # Dynamic guidance scale based on prediction difference
        if self.config.mode == "dynamic":
            with torch.no_grad():
                # Get predictions
                cond_pred = self.module(x, cond)
                uncond_pred = self.module(x, uncond)
                
                # Calculate difference
                diff = torch.abs(cond_pred - uncond_pred).mean()
                
                # Adjust guidance scale
                scale = self.config.guidance_scale
                if diff < self.config.dynamic_threshold:
                    scale = max(scale * 0.9, self.config.min_guidance)
                else:
                    scale = min(scale * 1.1, self.config.max_guidance)
        else:
            scale = self.config.guidance_scale
        
        # Memory optimization by caching
        if self.config.optimize_memory:
            if self.cached_cond is None or not torch.allclose(cond, self.cached_cond):
                self.cached_cond_out = self.module(x, cond)
                self.cached_cond = cond
            
            if self.cached_uncond is None or not torch.allclose(uncond, self.cached_uncond):
                self.cached_uncond_out = self.module(x, uncond)
                self.cached_uncond = uncond
            
            cond_out = self.cached_cond_out
            uncond_out = self.cached_uncond_out
        else:
            cond_out = self.module(x, cond)
            uncond_out = self.module(x, uncond)
        
        # Apply guidance scale
        return uncond_out + scale * (cond_out - uncond_out)

def optimize_with_cfg_zero(
    model: nn.Module,
    cfg_config: Optional[Dict[str, Any]] = None,
    mode: str = "dynamic",
    optimize_memory: bool = True,
    guidance_scale: float = 7.5,
    min_guidance: float = 1.0,
    max_guidance: float = 20.0,
    dynamic_threshold: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply CFG-Zero optimization to Wan2.1 model.
    
    Args:
        model: The model to optimize
        cfg_config: Optional CFG-Zero configuration
        mode: CFG-Zero mode ("dynamic" or "static")
        optimize_memory: Whether to optimize memory usage
        guidance_scale: Base guidance scale
        min_guidance: Minimum guidance scale for dynamic mode
        max_guidance: Maximum guidance scale for dynamic mode
        dynamic_threshold: Threshold for dynamic guidance adjustment
        target_modules: List of module names to apply optimization to
    
    Returns:
        Tuple of (optimized_model, metrics)
    """
    # Initialize metrics tracking
    metrics = OptimizationMetrics()
    metrics.start()
    
    # Create configuration
    config = CFGZeroConfig(
        mode=mode,
        optimize_memory=optimize_memory,
        guidance_scale=guidance_scale,
        min_guidance=min_guidance,
        max_guidance=max_guidance,
        dynamic_threshold=dynamic_threshold,
    ) if cfg_config is None else CFGZeroConfig.from_dict(cfg_config)
    
    # Create a new model to avoid modifying the original
    optimized_model = type(model)()
    optimized_model.load_state_dict(model.state_dict())
    
    # Replace target modules with CFG-Zero wrapper
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            if "diffusion" in name.lower():
                target_modules.append(name)
    
    for name in target_modules:
        parent_name = '.'.join(name.split('.')[:-1])
        module_name = name.split('.')[-1]
        parent = optimized_model if parent_name == '' else getattr(optimized_model, parent_name)
        
        # Get original module
        original_module = getattr(parent, module_name)
        
        # Replace with CFG-Zero wrapper
        setattr(parent, module_name, CFGZeroWrapper(original_module, config))
    
    # Stop metrics tracking and add config info
    metrics.stop()
    metrics.add_metric("config", config.to_dict())
    metrics.add_metric("num_optimized_modules", len(target_modules))
    
    return optimized_model, metrics.get_metrics() 