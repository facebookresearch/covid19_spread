#!/usr/bin/env python


import pickle
import os.path
import base64
from typing import List
import googleapiclient
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import tempfile
import nbformat
import os
from nbconvert import HTMLExporter, MarkdownExporter
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets.config import Config
from nb2mail import MailExporter


def email_notebook(
    notebook_pth: str, recipients: List[str], subject: str, exclude_input: bool = True
):
    creds = None
    cred_path = "/private/home/mattle/covid19_spread/.gmail_token.pickle"
    if not os.path.exists(cred_path):
        raise RuntimeError(f"GMail credentials do not exist!!!  Path: {cred_path}")
    with open(cred_path, "rb") as token:
        creds = pickle.load(token)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        try:
            with open(cred_path, "wb") as token:
                pickle.dump(creds, token)
        except Exception:
            pass

    discovery_doc = requests.get(
        "https://raw.githubusercontent.com/googleapis/google-api-go-client/master/gmail/v1/gmail-api.json"
    ).json()
    service = googleapiclient.discovery.build_from_document(
        discovery_doc, credentials=creds
    )
    # service = build('gmail', 'v1', credentials=creds)

    with open(notebook_pth, "r") as fin:
        nb = nbformat.read(fin, as_version=4)

    # exectute notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.path.dirname(notebook_pth)}})

    conf = Config()
    conf.MailExporter.exclude_input = exclude_input
    exporter = MailExporter(config=conf)

    nb["metadata"]["nb2mail"] = {
        "To": ", ".join(recipients),
        "From": "Covid 19 Forecasting Team <faircovid19group@gmail.com>",
        "Subject": subject,
    }

    (body, resources) = exporter.from_notebook_node(nb)
    req = (
        service.users()
        .messages()
        .send(
            userId="me",
            body={
                "raw": base64.urlsafe_b64encode(body.encode("utf-8")).decode("utf-8")
            },
        )
    )
    req.execute()


if __name__ == "__main__":
    user = os.environ["USER"]
    script_dir = os.path.dirname(os.path.realpath(__file__))
    email_notebook(
        f"{script_dir}/../notebooks/test.ipynb", [f"{user}@fb.com"], "Test email"
    )
