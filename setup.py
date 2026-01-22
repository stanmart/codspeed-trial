from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# C++ extension module
cpp_ext = Extension(
    "codspeed_trial.cpp_ext",
    sources=["src/codspeed_trial/cpp_ext.cpp"],
    include_dirs=[np.get_include()],
    language="c++",
    extra_compile_args=["-O3"],
)

# Cython extension module
cy_ext = Extension(
    "codspeed_trial.cy_ext",
    sources=["src/codspeed_trial/cy_ext.pyx"],
    extra_compile_args=["-O3"],
)

setup(
    ext_modules=[cpp_ext] + cythonize([cy_ext], language_level=3),
)
