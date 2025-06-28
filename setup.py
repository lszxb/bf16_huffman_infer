import platform
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

pkg_name = 'bf16_huffman_infer'


library_dirs = []

if platform.system() == 'Windows':
    cuda_home = os.getenv('CUDA_HOME', None)
    if cuda_home is None:
        cuda_home = os.path.join(os.path.dirname(sys.executable), 'Library')
    library_dirs.append(os.path.join(cuda_home, 'lib'))


setup(
    name=pkg_name,
    include_dirs=[],
    packages=find_packages(include=[
        pkg_name, f'{pkg_name}.*'
    ]),
    ext_modules=[
        CUDAExtension(
            f"{pkg_name}._C",
            [
                f"{pkg_name}/src/bf16_huffman_infer.cpp",
                f"{pkg_name}/src/kernel.cu",
            ],
            extra_compile_args = {
                'cxx':  ['-std=c++17', '-O3', '-DPy_LIMITED_API=0x03090000'],
                'nvcc': ['-std=c++17', '-arch=sm_75', '-O3', '--use_fast_math', 
                         '-lineinfo', '--ptxas-options=-v --warn-on-spills'],
            },
            library_dirs=library_dirs,
            py_limited_api=True,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
