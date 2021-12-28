from os import path
from setuptools import setup, find_packages


here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


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
      include_package_data=True)
