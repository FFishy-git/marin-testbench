# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate retained pooled moments and locally used outer factors against Torch and FP64."""

import torch

from .outer_diagonal_checks import fp64_diagonal_factors
from .upstream_full_model_checks import UpstreamProductVerifier


class OwnedOutputVerifier(UpstreamProductVerifier):
    """Keep original update checks while requiring exact coverage of each owned projection."""

    @torch.no_grad()
    def check_factors(self):
        factors = self.actual.factors()
        expected = {
            (layer, kind) for _, per_rank in self.fast.shape_groups for layer, kind, _ in per_rank[self.fast.rank]
        }
        if set(factors) != expected:
            raise ValueError("Owned output factors do not cover exactly the locally updated projections")
        for width, moment in self.actual.moment.items():
            torch.testing.assert_close(moment, self.reference.moment[width], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(moment.double(), self.fp64_moment[width], rtol=1e-5, atol=1e-6)
            exact = fp64_diagonal_factors(
                self.fp64_moment[width],
                self.actual.damping,
                self.actual.inner_power,
                self.actual.outer_power,
            )[1]
            for index, key in enumerate(self.actual.keys[width]):
                if key not in expected:
                    continue
                inner, outer = factors[key]
                if inner is not None:
                    raise ValueError("Owned outer-only output factors constructed an unused inner root")
                torch.testing.assert_close(outer, self.factors[key][1], rtol=0, atol=0)
                torch.testing.assert_close(outer.double(), exact[index], rtol=1e-5, atol=1e-6)
