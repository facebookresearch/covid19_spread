#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from setuptools import setup, find_packages

setup(
    name="covid19_spread",
    version="0.1",
    py_modules=["covid19_spread"],
    install_requires=["Click",],
    packages=find_packages(),
    entry_points="""
        [console_scripts]
        cv=cv:cli
        prepare-data=prepare_data:cli
        recurring=recurring:cli
    """,
)
