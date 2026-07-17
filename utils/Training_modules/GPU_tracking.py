import torch
def get_gpu_memory_mb(device):
    """Return GPU memory stats in MB matching nvidia-smi output."""
    if device.type != 'cuda':
        return {'allocated_mb': 0.0, 'reserved_mb': 0.0, 'max_reserved_mb': 0.0}
    
    device_id = device.index if device.index is not None else torch.cuda.current_device()
    
    # allocated = actual tensors (subset of reserved)
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
    
    # reserved = what nvidia-smi shows (includes CUDA context, caching allocator, etc.)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
    
    # max reserved since last reset
    max_reserved = torch.cuda.max_memory_reserved(device_id) / (1024 ** 2)
    
    return {
        'allocated_mb': round(allocated, 2),
        'reserved_mb': round(reserved, 2),        # This should match nvidia-smi
        'max_reserved_mb': round(max_reserved, 2)  # Peak usage
    }