# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project raw host-refresh event records into SwanLab-compatible scalar summaries."""

EVENT_FIELDS = ("input_refresh_installations", "output_refresh_installations")
REQUIRED_FIELDS = {"group", "index", "source_step", "installation_step", "installed"}


def scalar_history_row(row: dict) -> dict:
    """Preserve every original scalar and summarize separately retained structured events."""
    scalars = {key: value for key, value in row.items() if key not in EVENT_FIELDS}
    for field in EVENT_FIELDS:
        if field not in row:
            continue
        events = row[field]
        if not isinstance(events, list):
            raise ValueError("Host-refresh events must be a list")
        groups = {}
        for event in events:
            if (
                not isinstance(event, dict)
                or event.keys() != REQUIRED_FIELDS
                or not isinstance(event["group"], str)
                or not event["group"]
                or "/" in event["group"]
                or type(event["installed"]) is not bool
                or any(type(event[key]) is not int for key in ("index", "source_step", "installation_step"))
                or event["index"] < 0
                or event["source_step"] < 0
                or event["installation_step"] != row["step"] - 1
                or event["installation_step"] <= event["source_step"]
            ):
                raise ValueError("Invalid recorded host-refresh event")
            groups.setdefault(event["group"], []).append(event)
        totals = {"count": len(events), "installed": sum(event["installed"] for event in events)}
        totals["discarded"] = totals["count"] - totals["installed"]
        for group, records in groups.items():
            ages = [event["installation_step"] - event["source_step"] for event in records]
            installed = sum(event["installed"] for event in records)
            totals.update(
                {
                    f"{group}/count": len(records),
                    f"{group}/installed": installed,
                    f"{group}/discarded": len(records) - installed,
                    f"{group}/min_age": min(ages),
                    f"{group}/max_age": max(ages),
                    f"{group}/min_source_step": min(event["source_step"] for event in records),
                    f"{group}/max_source_step": max(event["source_step"] for event in records),
                }
            )
        scalars[field] = totals
    return scalars
