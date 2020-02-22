import os
import setuptools


def get_requirements():
    cwd = os.path.dirname(os.path.realpath(__file__))
    req_file = os.path.join(cwd, 'requirements.txt')
    with open(req_file, 'r') as fp:
        reqs = fp.read().splitlines()

    return reqs

    
setuptools.setup(
    name="pytorchDL",
    version="0.1",
    author="Milogav",
    description="Package containing network definitions and utilities for pytorch deep learning framework",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[get_requirements()]
)
