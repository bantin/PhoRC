from setuptools import setup, find_packages

setup(
	name='phorc',
	version='0.0.1',
	description='Photocurrent subtraction for dense neural circuit mapping',
	author='Benjamin Antin',
	author_email='benjaminantin1@gmail.com',
	packages=find_packages(),
	install_requires=[
		'numpy',
		'scipy',
		'scikit-learn',
        'h5py',
	],
)