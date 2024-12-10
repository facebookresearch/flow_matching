# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import os

import setuptools

NAME = "flow_matching"
DESCRIPTION = "Flow Matching for Generative Modeling"
URL = "https://github.com/facebookresearch/flow_matching"
EMAIL = "ylipman@meta.com"
# Alphabetical
AUTHOR = ",".join(
    [
        "Brian Karrer",
        "David Lopez-Paz",
        "Heli Ben-Hamu",
        "Itai Gat",
        "Marton Havasi",
        "Matthew Le",
        "Neta Shaul",
        "Peter Holderrieth",
        "Ricky T.Q. Chen",
        "Yaron Lipman",
    ]
)
REQUIRES_PYTHON = ">=3.9.0"

for line in open("flow_matching/__init__.py"):
    line = line.strip()
    if "__version__" in line:
        context = {}
        exec(line, context)
        VERSION = context["__version__"]

readme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")

try:
    with open(readme_path) as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(),
    extras_require={
        "dev": [
            "pre-commit",
            "black==22.6.0",
            "usort==1.0.4",
            "ufmt==2.3.0",
            "flake8==7.0.0",
            "pydoclint",
        ],
    },
    install_requires=["numpy", "torch", "torchdiffeq"],
    license="CC-by-NC",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
