#!/usr/bin/env python3

import os
import dropbox
from dropbox import DropboxOAuth2FlowNoRedirect
import datetime
import time
import json


class Uploader:
    def __init__(self, cache_loc=None):
        cache_loc = cache_loc or os.path.expanduser("~/.covid19/credentials.json")
        if not os.path.exists(cache_loc):
            raise ValueError("Missing credentials file!")
        cred = json.load(open(cache_loc))
        if "access_token" not in cred:
            cred["access_token"] = self.authorize(cred["app_key"], cred["app_secret"])
            json.dump(cred, open(cache_loc, "w"))

        self.client = dropbox.Dropbox(oauth2_access_token=cred["access_token"])

    def authorize(self, app_key, app_secret):
        auth_flow = DropboxOAuth2FlowNoRedirect(app_key, app_secret)
        authorize_url = auth_flow.start()
        print("1. Go to: " + authorize_url)
        print('2. Click "Allow" (you might have to log in first).')
        print("3. Copy the authorization code.")
        auth_code = input("Enter the authorization code here: ").strip()

        try:
            oauth_result = auth_flow.finish(auth_code)
        except Exception as e:
            print("Error: %s" % (e,))
            sys.exit(1)
        return oauth_result.access_token

    def upload(self, source_file, dest):
        path = os.path.join("/covid19_forecasts", dest)
        mode = dropbox.files.WriteMode.overwrite
        mtime = os.path.getmtime(source_file)
        with open(source_file, "rb") as f:
            data = f.read()
        try:
            res = self.client.files_upload(
                data,
                path,
                mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True,
            )
        except dropbox.exceptions.ApiError as err:
            print("*** API error", err)
            return None
        print("uploaded as", res.name.encode("utf8"))
        return res
