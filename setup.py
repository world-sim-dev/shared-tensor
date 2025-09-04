#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="shared-tensor",
    version="0.1.0",
    description="A library for sharing GPU memory objects across processes using IPC mechanisms",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/world-sim-dev/shared-tensor",
    packages=find_packages(exclude=["tests", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware :: Symmetric Multi-processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "pytest-benchmark>=3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "shared-tensor-server=shared_tensor.server:main",
        ],
    },
    package_data={
        "shared_tensor": [
            "*.so",
            "*.dll",
            "*.dylib",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "gpu",
        "memory",
        "sharing",
        "ipc",
        "inter-process-communication",
        "pytorch",
        "tensorflow",
        "cuda",
        "model-serving",
        "inference",
        "distributed-computing",
    ],
    author="Athena Team",
    author_email="contact@world-sim-dev.org",
    project_urls={
        "Bug Reports": "https://github.com/world-sim-dev/shared-tensor/issues",
        "Source": "https://github.com/world-sim-dev/shared-tensor",
        "Documentation": "https://github.com/world-sim-dev/shared-tensor/wiki",
    },
)
