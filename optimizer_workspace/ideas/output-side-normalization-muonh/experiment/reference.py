# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the preserved Toner snapshot without modifying it."""

import sys
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REVISION = "dab7f3777c75e740536df822f50ffa1d929a340c"
UPSTREAM = TASK / "artifacts" / "external-sources" / f"Toner-{REVISION}"
sys.path.insert(0, str(UPSTREAM / "src"))
sys.path.insert(0, str(UPSTREAM))
