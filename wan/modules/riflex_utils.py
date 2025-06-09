import torch
import torch.cuda.amp as amp

@amp.autocast(enabled=False)
def rope_params_riflex(max_seq_len, dim, theta=10000, k=None, L_test=None):
    """
    Modified RoPE parameters with RIFLEX's frequency adjustment.
    
    Args:
        max_seq_len: Maximum sequence length
        dim: Dimension of the embeddings
        theta: Base for the exponential
        k: Index for the intrinsic frequency in RoPE
        L_test: Number of frames for inference
    """
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    
    # RIFLEX modification: Reduce intrinsic frequency to stay within a single period
    if k is not None and L_test is not None:
        # Multiply by 0.9 to keep extrapolated length below 90% of a period
        freqs[:, k-1] = 0.9 * 2 * torch.pi / L_test
    
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs 