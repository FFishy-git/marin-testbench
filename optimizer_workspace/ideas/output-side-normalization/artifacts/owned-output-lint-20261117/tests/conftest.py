# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

task = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(task))
revision = json.loads((task / "experiment/upstream_manifest.json").read_text())["revision"]
upstream = task / "artifacts/external-sources" / f"Toner-{revision}"
sys.path.insert(0, str(upstream))
sys.path.insert(0, str(upstream / "src"))
