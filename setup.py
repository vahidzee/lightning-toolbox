from setuptools import setup, find_packages


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lightning_toolbox",
    packages=find_packages(include=["lightning_toolbox", "lightning_toolbox.*"]),
    version="0.0.7",
    license="MIT",
    description="A collection of utilities for PyTorch Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vahid Zehtab",
    author_email="vahid@zehtab.me",
    url="https://github.com/vahidzee/lightning_toolbox",
    keywords=["artificial intelligence", "pytorch lightning", "objective functions", "regularization"],
    install_requires=["torch>=1.9", "lightning", "dycode==0.0.2"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
