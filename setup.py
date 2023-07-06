from setuptools import find_packages, setup

setup(
    name='ESQmodel',
    version='0.1.0',
    description='A biologically informed evaluation of 2-D cell segmentation quality in multiplexed tissue images.',
    author='Eric Lee',
    author_email='psylin223@gmail.com',
    url='https://github.com/ericlee0920/IMC-QC-Model',
    packages=find_packages(),
    license='MIT',
    entry_points={
        'console_scripts': [
            'esq=src.cli:main',
        ]
    }
)
