# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deny execution until a separate Wave11 production release is implemented."""


def require_selected_lr_production_owner(args: object) -> None:
    raise RuntimeError(
        "Wave11 full-budget production requires a separate generation-bound production release; "
        "this preparation unconditionally denies GPU execution"
    )
