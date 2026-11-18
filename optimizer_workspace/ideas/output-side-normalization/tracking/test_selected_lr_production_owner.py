# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the production denial both in tracking and in a copied candidate."""

from types import SimpleNamespace

import pytest
from selected_lr_production_owner import require_selected_lr_production_owner


def test_selected_lr_production_denies_claimed_release_without_creating_training_output(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    release = tmp_path / "wave11-launch-binding.json"
    release.write_text('{"status":"frozen_validated_launch_release"}\n')
    args = SimpleNamespace(
        config=tmp_path / "unread-config.json",
        out=tmp_path / "uncreated-run",
        max_steps=24,
        execution_boundary=22,
        launch_release=release,
    )
    with pytest.raises(RuntimeError, match="separate generation-bound production release"):
        require_selected_lr_production_owner(args)
    assert not args.out.exists()
    assert not args.config.exists()
