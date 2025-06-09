import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import math

from . import BaseConfig, OptimizationMetrics

__all__ = ['optimize_with_transformer_quant', 'TRANSFORMER_QUANT_AVAILABLE']

TRANSFORMER_QUANT_AVAILABLE = True  # Since this is our own implementation

class TransformerQuantConfig(BaseConfig):
    """Configuration for transformer quantization."""
    def __init__(
        self,
        bits: int = 8,
        method: str = "symmetric",
        per_channel: bool = True,
        calibration_steps: int = 100,
        dynamic_range: bool = True,
    ):
        super().__init__()
        self.bits = bits
        self.method = method
        self.per_channel = per_channel
        self.calibration_steps = calibration_steps
        self.dynamic_range = dynamic_range

class QuantizedModule(nn.Module):
    """Wrapper for quantized transformer modules."""
    def __init__(self, module: nn.Module, config: TransformerQuantConfig):
        super().__init__()
        self.module = module
        self.config = config
        self.scale = None
        self.zero_point = None
        self.calibrated = False
    
    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate quantization parameters."""
        if self.calibrated:
            return
            
        with torch.no_grad():
            if self.config.method == "symmetric":
                max_val = torch.max(torch.abs(x))
                self.scale = (2 ** (self.config.bits - 1) - 1) / max_val
                self.zero_point = 0
            else:  # asymmetric
                min_val = torch.min(x)
                max_val = torch.max(x)
                self.scale = (2 ** self.config.bits - 1) / (max_val - min_val)
                self.zero_point = -min_val * self.scale
                
        self.calibrated = True
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor."""
        if not self.calibrated:
            self.calibrate(x)
            
        x_scaled = x * self.scale + self.zero_point
        x_clipped = torch.clamp(x_scaled, 0, 2 ** self.config.bits - 1)
        x_rounded = torch.round(x_clipped)
        return (x_rounded - self.zero_point) / self.scale
    
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass with quantization."""
        if self.config.dynamic_range:
            self.calibrated = False  # Recalibrate for dynamic quantization
            
        x = self.quantize(x)
        return self.module(x, *args, **kwargs)

def optimize_with_transformer_quant(
    model: nn.Module,
    quant_config: Optional[Dict[str, Any]] = None,
    bits: int = 8,
    method: str = "symmetric",
    per_channel: bool = True,
    calibration_steps: int = 100,
    dynamic_range: bool = True,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply transformer quantization optimization to Wan2.1 model.
    
    Args:
        model: The model to optimize
        quant_config: Optional quantization configuration
        bits: Number of quantization bits
        method: Quantization method ("symmetric" or "asymmetric")
        per_channel: Whether to use per-channel quantization
        calibration_steps: Number of steps for calibration
        dynamic_range: Whether to use dynamic quantization
        target_modules: List of module names to apply optimization to
    
    Returns:
        Tuple of (optimized_model, metrics)
    """
    # Initialize metrics tracking
    metrics = OptimizationMetrics()
    metrics.start()
    
    # Create configuration
    config = TransformerQuantConfig(
        bits=bits,
        method=method,
        per_channel=per_channel,
        calibration_steps=calibration_steps,
        dynamic_range=dynamic_range,
    ) if quant_config is None else TransformerQuantConfig.from_dict(quant_config)
    
    # Create a new model to avoid modifying the original
    optimized_model = type(model)()
    optimized_model.load_state_dict(model.state_dict())
    
    # Replace target modules with quantized versions
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                target_modules.append(name)
    
    for name in target_modules:
        parent_name = '.'.join(name.split('.')[:-1])
        module_name = name.split('.')[-1]
        parent = optimized_model if parent_name == '' else getattr(optimized_model, parent_name)
        
        # Get original module
        original_module = getattr(parent, module_name)
        
        # Replace with quantized wrapper
        setattr(parent, module_name, QuantizedModule(original_module, config))
    
    # Stop metrics tracking and add config info
    metrics.stop()
    metrics.add_metric("config", config.to_dict())
    metrics.add_metric("num_optimized_modules", len(target_modules))
    
    return optimized_model, metrics.get_metrics() 