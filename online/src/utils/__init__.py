import os
import site
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