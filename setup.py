import platform
import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

pkg_name = 'bf16_huffman_infer'


os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '7.5+PTX')


library_dirs = []

if platform.system() == 'Windows':
    cuda_home = os.getenv('CUDA_HOME', None)
    if cuda_home is None:
        cuda_home = os.path.join(os.path.dirname(sys.executable), 'Library')
    library_dirs.append(os.path.join(cuda_home, 'lib'))


setup(
    name=pkg_name,
    version='0.0.1',
    author='lszxb',
    description='Fused BF16 Huffman GEMV Inference kernel',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lszxb/bf16_huffman_infer',
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
                'cxx':  ['-std=c++17', '-O2', '-DPy_LIMITED_API=0x03090000'],
                'nvcc': ['-std=c++17', '-O3', '--use_fast_math', 
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
