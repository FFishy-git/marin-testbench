# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record authorized source access and resolve full dataset revisions."""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from .records import atomic_json, sha256, utc_now
from .reference import UPSTREAM


def probe(proxy):
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy
    plan = json.loads((UPSTREAM / "reproduce/marin/data_plan.json").read_text())
    token = get_token()
    api = HfApi(token=token)
    sources = {}
    for source in plan["sources"].values():
        if source["repo"] in sources:
            continue
        try:
            info = api.dataset_info(source["repo"], revision=source["revision"])
            sources[source["repo"]] = {
                "recorded_revision": source["revision"],
                "resolved_revision": info.sha,
                "accessible_metadata": True,
                "gated": info.gated,
            }
        except HfHubHTTPError as error:
            sources[source["repo"]] = {
                "recorded_revision": source["revision"],
                "accessible_metadata": False,
                "http_status": error.response.status_code,
            }
    tokenizer = {
        "repo": plan["tokenizer"],
        "revision": plan["tokenizer_revision"],
        "accessible": False,
    }
    try:
        path = hf_hub_download(plan["tokenizer"], "tokenizer.json", revision=plan["tokenizer_revision"], token=token)
        tokenizer.update(accessible=True, tokenizer_sha256=sha256(Path(path)), cache_path=path)
    except HfHubHTTPError as error:
        tokenizer["http_status"] = error.response.status_code
    return {
        "time": utc_now(),
        "proxy": proxy,
        "hf_authorization_configured": token is not None,
        "datasets": sources,
        "tokenizer": tokenizer,
        "limitation": "Dataset metadata access does not establish access to all source files.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default="http://star-proxy.oa.com:3128")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = probe(args.proxy)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
