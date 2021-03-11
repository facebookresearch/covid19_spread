#!/usr/bin/env python3
# Copyright (c) 2021-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from slack import WebClient
import os
import json


def get_client():
    credentials = json.load(open(os.path.expanduser("~/.credentials.json")))["slack"]
    client = WebClient(token=credentials["bot_token"])
    return client
