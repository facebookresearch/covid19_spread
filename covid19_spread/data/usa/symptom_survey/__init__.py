#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .fetch import main as fetch, SIGNALS
from .process_symptom_survey import main as process


def prepare():
    for source, signal in SIGNALS:
        fetch("state", source, signal)
        fetch("county", source, signal)
        process(f"{source}/{signal}", "state")
        process(f"{source}/{signal}", "county")
