# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the selected-LR stage-profile preparation GPU denied."""


def require_selected_lr_stage_profile_owner(args: object) -> None:
    raise RuntimeError(
        "Selected-LR stage profiling requires a separate generation-bound profiling release; "
        "this preparation unconditionally denies GPU execution"
    )
