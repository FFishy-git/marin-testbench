# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serializable numerical state of a Toner fast engine."""

import concurrent.futures

from toner.fast import roots as PR


def engine_state(engine):
    """Save numerical root state, including cached bases and pending refreshes."""
    engine.side.synchronize()
    for group in engine.groups.values():
        for pending in group.pending:
            if pending is not None:
                pending[0].result()
    groups = {}
    for name, group in engine.groups.items():
        groups[name] = {
            "C": group.C,
            "tol": group.tol,
            "basis_step": group.basis_step,
            "warm": group.warm,
            "cached_root_graph_active": group.graph is not None,
            "SB": None if group.basis is None else group.basis.SB,
            "S2": None if group.basis is None else group.basis.S2,
            "pending": [
                (
                    None
                    if pending is None
                    else {
                        "step": pending[1],
                        "basis": group.host_out[index].clone(),
                        "tol": group.host_tol[index].clone(),
                    }
                )
                for index, pending in enumerate(group.pending)
            ],
        }
    return {
        "groups": groups,
        "momentum": engine.state,
        "weight": engine.weight,
        "steps": engine.steps,
        "t": engine.t,
        "fallbacks": engine.fallbacks,
        "extra": engine.extra,
        "refresh_install_age": engine.refresh_install_age,
    }


def restore_engine(engine, state):
    if state["refresh_install_age"] != engine.refresh_install_age:
        raise ValueError("Checkpoint refresh-installation policy differs")
    for name, saved in state["groups"].items():
        group = engine.groups[name]
        group.C.copy_(saved["C"])
        group.tol.copy_(saved["tol"])
        group.basis_step = saved["basis_step"]
        # CUDA graphs and process-local kernel compilation are not serializable.
        # Rewarm against the restored numerical state before capturing fresh graphs.
        group.warm = False
        if saved["S2"] is not None:
            group.basis = PR.Basis.of_splits(saved["S2"].to(engine.device))
            group.basis.SB = saved["SB"].to(engine.device)
        for index, pending in enumerate(saved["pending"]):
            if pending is not None:
                group.host_out[index].copy_(pending["basis"])
                group.host_tol[index].copy_(pending["tol"])
                future = concurrent.futures.Future()
                future.set_result(None)
                group.pending[index] = (future, pending["step"])
    engine.state = {
        shape: {name: tensor.to(engine.device) for name, tensor in values.items()}
        for shape, values in state["momentum"].items()
    }
    engine.weight, engine.steps, engine.t = state["weight"], state["steps"], state["t"]
    engine.fallbacks, engine.extra = state["fallbacks"], state["extra"]
