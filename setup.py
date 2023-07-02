from setuptools import setup, find_packages

setup(
	name='phorc',
	version='0.0.1',
	description='Photocurrent subtraction for optogenetic connectivity mapping data',
	author='Benjamin Antin',
	author_email='benjaminantin1@gmail.com',
	packages=find_packages(),
	install_requires=[
        'h5py',
	],
)