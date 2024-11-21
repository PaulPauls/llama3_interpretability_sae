import torch


def set_up_cuda():
    """Reset memory stats and set up CUDA environment."""
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def set_torch_seed_for_inference(seed: int):
    """Set all seeds to make results reproducible for inference."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
