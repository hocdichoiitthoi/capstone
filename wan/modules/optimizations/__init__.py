"""
Optimizations module for Wan2.1.
This module integrates various optimizations:
- SageAttention 2: Optimized attention with INT8 quantization
- Transformer Quantization: Dynamic quantization for transformers
- CFG-Zero-star: Zero-shot CFG optimization
- TeaCache: Memory-efficient caching system
"""

import torch
import time
from typing import Dict, Any, Optional
import psutil
import GPUtil

class OptimizationMetrics:
    """Common metrics tracking for all optimizations."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.start_cpu_memory = None
        self.end_cpu_memory = None
        self.start_gpu_memory = None
        self.end_gpu_memory = None
        self.additional_metrics = {}
    
    def start(self):
        """Start tracking metrics."""
        self.start_time = time.time()
        self.start_cpu_memory = psutil.Process().memory_info().rss
        self.start_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    def stop(self):
        """Stop tracking metrics."""
        self.end_time = time.time()
        self.end_cpu_memory = psutil.Process().memory_info().rss
        self.end_gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric."""
        self.additional_metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all tracked metrics."""
        if not all([self.start_time, self.end_time, 
                   self.start_cpu_memory, self.end_cpu_memory,
                   self.start_gpu_memory, self.end_gpu_memory]):
            raise RuntimeError("Metrics tracking not properly started/stopped")
        
        return {
            "execution_time": self.end_time - self.start_time,
            "cpu_memory_used_mb": (self.end_cpu_memory - self.start_cpu_memory) / (1024 * 1024),
            "gpu_memory_used_mb": (self.end_gpu_memory - self.start_gpu_memory) / (1024 * 1024),
            **self.additional_metrics
        }

class BaseConfig:
    """Base configuration class for all optimizations."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create config from dictionary."""
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return config

# Import optimization modules
from .riflex import optimize_with_riflex, RIFLEX_AVAILABLE
from .cfg_zero import optimize_with_cfg_zero, CFG_ZERO_AVAILABLE
from .transformer_quant import optimize_with_transformer_quant, QUANT_AVAILABLE
from .tea_cache import optimize_with_tea_cache, TEA_CACHE_AVAILABLE
from .sage_attention import optimize_with_sage, SAGE_ATTN_AVAILABLE

__all__ = [
    'OptimizationMetrics',
    'BaseConfig',
    'optimize_with_riflex',
    'RIFLEX_AVAILABLE',
    'optimize_with_cfg_zero',
    'CFG_ZERO_AVAILABLE',
    'optimize_with_transformer_quant',
    'QUANT_AVAILABLE',
    'optimize_with_tea_cache',
    'TEA_CACHE_AVAILABLE',
    'optimize_with_sage',
    'SAGE_ATTN_AVAILABLE',
] 