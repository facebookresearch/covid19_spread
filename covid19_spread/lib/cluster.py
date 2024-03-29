#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import getpass


USER = getpass.getuser()
if os.path.exists(f"/checkpoint"):
    FS = "/checkpoint"
    PARTITION = "learnfair"
    MEM_GB = lambda x: x
elif os.path.exists(f"/fsx"):
    FS = "/fsx"
    PARTITION = "compute"
    MEM_GB = lambda x: 0
else:
    FS = os.getcwd()  # for CI
