from setuptools import setup, find_packages

setup(
   name='aif-pointing',
   version='1.0',
   author='Markus Klar',
   author_email='markus.klar@glasgow.ac.uk',
   packages=['difai-base'],
   package_data={'': ['**']},
   license='LICENSE',
   python_requires='>=3.11',
   install_requires=[
       "numpy"
   ]
)