# Copyright Vahid Zehtab (vahid@zehtab.me) 2021
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import setup, find_packages


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="lightning_toolbox",
    packages=find_packages(include=["lightning_toolbox", "lightning_toolbox.*"]),
    version="0.0.19",
    license="MIT",
    description="A collection of utilities for PyTorch Lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vahid Zehtab",
    author_email="vahid@zehtab.me",
    url="https://github.com/vahidzee/lightning-toolbox",
    keywords=["artificial intelligence", "pytorch lightning", "objective functions", "regularization"],
    install_requires=["torch>=1.9", "lightning>=1.9.0", "dypy"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": ["lightning_toolbox=lightning_toolbox.main:main"],
    },
)
