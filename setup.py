from setuptools import setup, find_packages

setup(
    name="DLpy",  # Changed from DLpy to DLpy
    version="0.1.0",
    author="Antonio Lujano Luna",
    packages=find_packages(include=["DLpy"]), 
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    python_requires=">=3.8",
)