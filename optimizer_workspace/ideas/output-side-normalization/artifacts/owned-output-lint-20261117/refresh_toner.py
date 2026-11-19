# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Install host-refreshed bases at a fixed optimizer-update age."""

import torch
from toner.fast.engine import FastTonerCore


def join_refreshes(group, *, step, age):
    """Join eligible decompositions without installing early-completed snapshots."""
    eligible = [
        (index, pending[1])
        for index, pending in enumerate(group.pending)
        if pending is not None and step - pending[1] >= age
    ]
    for index, _ in eligible:
        group.pending[index][0].result()
    return eligible


def install_refreshes(group, *, step, age, device):
    """Install eligible nonstale snapshots and report the actual update boundary."""
    ready = join_refreshes(group, step=step, age=age)
    if not ready:
        return []
    mask = [False] * group.per
    records = []
    for index, source_step in ready:
        installed = source_step >= group.basis_step[index]
        if installed:
            group.stage[index].copy_(group.host_out[index], non_blocking=True)
            group.tol[index].copy_(group.host_tol[index], non_blocking=True)
            mask[index] = True
            group.basis_step[index] = source_step
        records.append(
            {
                "index": index,
                "source_step": source_step,
                "installation_step": step,
                "installed": installed,
            }
        )
        group.pending[index] = None
    if any(mask):
        installation_mask = torch.tensor(mask)
        if torch.device(device).type == "cuda":
            installation_mask = installation_mask.pin_memory()
        group.basis.install_many(group.stage, installation_mask.to(device, non_blocking=True))
    return records


class RefreshTonerCore(FastTonerCore):
    """The pinned numerical engine with deterministic host-refresh installation."""

    def __init__(self, *args, refresh_install_age, **kwargs):
        if type(refresh_install_age) is not int or refresh_install_age < 1:
            raise ValueError("Refresh installation age must be a positive integer")
        self.refresh_install_age = refresh_install_age
        self.refresh_installations = []
        super().__init__(*args, **kwargs)

    def _install_ready(self):
        self.refresh_installations = []
        for name, group in self.groups.items():
            installed = install_refreshes(group, step=self.steps, age=self.refresh_install_age, device=self.device)
            self.refresh_installations.extend({"group": name, **record} for record in installed)
