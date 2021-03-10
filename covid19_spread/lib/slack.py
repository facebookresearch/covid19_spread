#!/usr/bin/env python3

from slack import WebClient
import os
import json


def get_client():
    credentials = json.load(open(os.path.expanduser("~/.credentials.json")))["slack"]
    client = WebClient(token=credentials["bot_token"])
    return client
