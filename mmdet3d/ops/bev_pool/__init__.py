import os
import torch
from torch.utils import cpp_extension
from .bev_pool import bev_pool
cwd = os.path.dirname(os.path.realpath(__file__))

sources = []

if torch.cuda.is_available():
    sources.append(os.path.join(cwd, 'src', 'bev_pool.cpp'))
    sources.append(os.path.join(cwd, 'src', 'bev_pool_cuda.cu'))

extra_cuda_cflags=[
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__"
]

bev_pool_ext = None

# Try pre-built .so first
_so_path = os.path.join(cwd, 'bev_pool_ext.so')
if os.path.exists(_so_path):
    import importlib
    try:
        bev_pool_ext = importlib.machinery.ExtensionFileLoader(
            'bev_pool_ext', _so_path).load_module()
    except Exception:
        pass

if bev_pool_ext is None:
    bev_pool_ext = cpp_extension.load('bev_pool_ext',
                                    sources=sources,
                                    build_directory=cwd,
                                    extra_cuda_cflags=extra_cuda_cflags,
                                    verbose=False)
