#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from .process_mobility import main as mobility_main
from .process_open_data import main as open_data_main


def prepare():
    mobility_main()
    open_data_main()
