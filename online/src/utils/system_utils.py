import os
import site
import jax
import os
import psutil
import gc
import torch
import jax

def setup_environment():
    """Applies necessary PTXAS and CUDA fixes for JAX/Flax."""
    # PTXAS FIX
    for path in site.getsitepackages():
        ptxas_path = os.path.join(path, "nvidia/cuda_nvcc/bin")
        if os.path.exists(ptxas_path):
            os.environ["PATH"] = ptxas_path + ":" + os.environ.get("PATH", "")
            break

    # EXISTING CUDA FIXES
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        cudnn_path = os.path.join(conda_prefix, "lib/python3.12/site-packages/nvidia/cudnn/lib")
        if os.path.exists(cudnn_path):
            os.environ["LD_LIBRARY_PATH"] = cudnn_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def calculate_grid_memory_mb(res, features=36, dtype_size=2):
    return (res**3 * features * dtype_size) / (1024**2)

def get_vram_usage_mb():
    try:
        device = jax.devices()[0]
        stats = device.memory_stats()
        used_mb = stats['bytes_in_use'] / (1024**2)
        peak_mb = stats.get('peak_bytes_in_use', stats['bytes_in_use']) / (1024**2)
        return used_mb, peak_mb
    except:
        return 0.0, 0.0

def get_torch_vram_usage_mb():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0.0

class MemoryTracker:
    def __init__(self, device_type='cuda'):
        self.device_type = device_type
        self.process = psutil.Process(os.getpid())
        self.start_ram = 0
        self.peak_vram_mb = 0.0
        self.peak_ram_mb = 0.0
        
    def start(self):
        """Reset stats and record baseline."""
        gc.collect()
        
        # 1. CPU Baseline
        self.start_ram = self.process.memory_info().rss / (1024**2)
        
        # 2. GPU Baseline (Reset Peak Tracker)
        if self.device_type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def get_current_usage(self):
        """Returns (vram_mb, ram_mb) currently in use."""
        ram_mb = self.process.memory_info().rss / (1024**2)
        vram_mb = 0.0
        
        if self.device_type == 'cuda' and torch.cuda.is_available():
            vram_mb = torch.cuda.memory_reserved() / (1024**2)
            
        return vram_mb, ram_mb

    def stop_and_report(self):
        """Returns peak usage since start()."""
        current_ram = self.process.memory_info().rss / (1024**2)
        peak_ram_delta = max(0, current_ram - self.start_ram)
        
        peak_vram = 0.0
        if self.device_type == 'cuda' and torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_reserved() / (1024**2)
            
        return {
            "peak_vram_mb": peak_vram,
            "peak_ram_mb": peak_ram_delta 
        }