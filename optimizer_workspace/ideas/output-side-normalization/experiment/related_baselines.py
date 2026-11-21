# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit routing for controlled and source-recipe related optimizer arms."""

from .go_muon import GoMuon
from .mousse import Mousse
from .related_base import MuonControl

RELATED_ARMS = {
    "muon_control": MuonControl,
    "go_muon_control": GoMuon,
    "go_muon_paper": GoMuon,
    "mousse_control": Mousse,
    "mousse_paper": Mousse,
}


def build_related(model, config, **runtime):
    return RELATED_ARMS[config["experiment"]["arm"]](model, config, **runtime)
