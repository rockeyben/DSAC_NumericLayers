from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(Extension("NumericLayers",
    						language="c++",
                             sources=["NumericLayers.pyx", "c_NumericLayers.cpp"],
                             include_dirs=[numpy.get_include()],
                             extra_compile_args=["-fopenmp"],
                             extra_link_args=["-L/usr/local/lib","-lopencv_core", "-lopencv_imgcodecs","-lopencv_highgui", "-lopencv_calib3d", "-fopenmp"],))
)
