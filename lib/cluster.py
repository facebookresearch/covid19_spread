#!/usr/bin/env python3

import os

USER = os.environ["USER"]
if os.path.exists(f"/checkpoint"):
    FS = "/checkpoint"
    PARTITION = "learnfair"
    MEM_GB = lambda x: x
elif os.path.exists(f"/fsx"):
    FS = "/fsx"
    PARTITION = "compute"
    MEM_GB = lambda x: 0
else:
    print("WARNINING: cannot determine distributed file system")
