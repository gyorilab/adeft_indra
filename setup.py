import sys
import numpy as np
from os import path
from setuptools.extension import Extension
from setuptools import dist, setup, find_packages, Command


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


if '--use-cython' in sys.argv:
    USE_CYTHON = True
    sys.argv.remove('--use-cython')
else:
    USE_CYTHON = False

ext = '.pyx' if USE_CYTHON else '.c'

defs = [('NPY_NO_DEPRECATED_API', 0)]
inc_path = np.get_include()
lib_path = path.join(inc_path, '..', '..', 'random', 'lib')

extensions = [
    Extension('adeft_indra.anomaly_detection.stats._stats',
              ['adeft_indra/anomaly_detection/stats/_stats' + ext],
              include_dirs=[inc_path],
              library_dirs=[lib_path],
              libraries=['npyrandom'],
              define_macros=defs)
    ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions,
                           compiler_directives={'language_level': 3})

setup(name='adeft_indra',
      version='0.0.0',
      description='Adeft model building pipeline.',
      author='adeft developers, Harvard Medical School',
      author_email='albert_steppi@hms.harvard.edu',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8'
      ],
      packages=find_packages(),
      ext_modules=extensions,
      include_package_data=True)
