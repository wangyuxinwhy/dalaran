import gc
from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def use_dtype(dtype: torch.dtype):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def is_distributed() -> bool:
    try:
        return torch.distributed.is_initialized()  # type: ignore
    except AttributeError:
        return False


def is_main_process() -> bool:
    try:
        return torch.distributed.get_rank() == 0  # type: ignore
    except AttributeError:
        return True


def clear_memory(reset_cuda_stats: bool = True) -> None:
    """
    Clears the memory by performing garbage collection and emptying the CUDA cache (if available).

    Args:
        reset_cuda_stats (bool): Whether to reset CUDA memory statistics. Default is True.

    Returns:
        None
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_cuda_stats:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()


def only_in_main_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            result = func(*args, **kwargs)
        torch.distributed.barrier()  # type: ignore
        return result

    return wrapper
