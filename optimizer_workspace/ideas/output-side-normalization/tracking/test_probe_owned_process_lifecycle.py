# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check retained lifecycle evidence, output isolation and real timeout cleanup."""

import hashlib
import json
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

SOURCE = Path(__file__).with_name("probe_owned_process_lifecycle.py")


def run_cli(out: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(SOURCE), "--out", str(out), *args],
        env={"UNRELATED_TEST_VALUE": "not-to-be-retained"},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_lifecycle_cli_retains_redacted_controls_and_reaps_all_children(tmp_path):
    out = tmp_path / "retained"
    result = run_cli(out)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    evidence = json.loads((out / "evidence.json").read_text())
    assert receipt["status"] == "completed"
    assert receipt["cleanup"]["remaining_children"] == []
    assert receipt["started"]["unix_ns"] < receipt["finished"]["unix_ns"]
    assert receipt["kernel"]["release"]
    assert receipt["kernel"]["boot_id"]
    for artifact in receipt["artifacts"].values():
        payload = Path(artifact["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == artifact["sha256"]
        assert len(payload) == artifact["bytes"]
        assert b"not-to-be-retained" not in payload
    assert (out / SOURCE.name).read_bytes() == SOURCE.read_bytes()
    controls = evidence["controls"]
    exits = controls["exit_teardown"]
    assert exits["mapping_count"] == 8192
    for attempt in exits["attempts"]:
        terminal = attempt["terminal"]
        assert terminal["waitid"]["si_status"] == terminal["returncode"] == 37
        assert terminal["proc_absent_after_reap"]
        assert attempt["old_environment_fd_after_exit_bytes"] == 0
        teardown = attempt["nonterminal_teardown"]
        if teardown is not None:
            assert teardown["command"]["byte_count"] == teardown["environment"]["byte_count"] == 0
            for bracket in ("stat_before", "stat_after"):
                assert teardown[bracket]["pf_exiting"]
                assert teardown[bracket]["vsize"] == 0
                assert teardown[bracket]["status"] not in ("Z", "X", "x")
    assert exits["pre_zombie_teardown_observed"] == any(
        attempt["nonterminal_teardown"] is not None for attempt in exits["attempts"]
    )
    variants = controls["exec_marker_removal"]["variants"]
    for variant in variants:
        before, after = variant["before"], variant["after"]
        assert before["stat_after"]["pid"] == after["stat_after"]["pid"]
        assert before["stat_after"]["start_ticks"] == after["stat_after"]["start_ticks"]
        assert after["stat_after"]["status"] not in ("Z", "X", "x")
        assert not after["stat_after"]["pf_exiting"]
        assert not after["environment"]["marker_present"]
        assert variant["old_environment_fd_after_exec_bytes"] == 0
        assert variant["terminal"]["returncode"] == 0
        assert variant["terminal"]["proc_absent_after_reap"]
        pidfd = variant["pidfd"]
        assert pidfd["opened"]["syscall_number"] == 434
        assert pidfd["opened"]["stat_at_open"]["stat_after"]["status"] not in ("Z", "X", "x")
        assert not pidfd["initial_leader_live"]["readable"]
        assert not pidfd["replacement_live"]["readable"]
        assert variant["terminal"]["pidfd_before_reap"]["readable"]
        assert pidfd["opened"]["descriptor"] == variant["terminal"]["pidfd_before_reap"]["descriptor"]
        for rescan in variant["complete_rescans_after_exec"]:
            assert rescan["members_before"] == rescan["members_after"] == [after["stat_after"]["pid"]]
            assert not rescan["processes"][0]["sample"]["stat_after"]["pf_exiting"]
    leader, nonleader = variants
    assert leader["before"]["environment"]["marker_present"]
    assert leader["before"]["command"]["sha256"] != leader["after"]["command"]["sha256"]
    leader_exit = nonleader["leader_exited_with_live_thread"]["processes"][0]
    assert leader_exit["sample"]["stat_after"]["pf_exiting"]
    assert leader_exit["sample"]["stat_after"]["status"] == "Z"
    assert not nonleader["pidfd"]["zombie_leader_live_worker"]["readable"]
    assert any(
        thread["sample"]["environment"].get("marker_present") and not thread["sample"]["stat_after"]["pf_exiting"]
        for thread in leader_exit["threads"]
    )
    reparent = controls["reparenting"]
    before, after = reparent["before"], reparent["after"]
    assert before["stat_after"]["parent"] != after["stat_after"]["parent"]
    for field in ("pid", "start_ticks", "group", "session"):
        assert before["stat_after"][field] == after["stat_after"][field]
    assert before["command"] == after["command"]
    assert before["environment"] == after["environment"]
    assert after["environment"]["marker_present"]
    assert not after["stat_after"]["pf_exiting"]
    for terminal in evidence["reaped_children"]:
        assert terminal["proc_absent_after_reap"]
        assert not (Path("/proc") / str(terminal["pid"])).exists()


def test_lifecycle_cli_refuses_existing_output_without_overwriting(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    receipt = out / "receipt.json"
    receipt.write_bytes(b"retained-existing-evidence")
    result = run_cli(out)
    assert result.returncode != 0
    assert receipt.read_bytes() == b"retained-existing-evidence"
    assert sorted(path.name for path in out.iterdir()) == ["receipt.json"]


def test_lifecycle_cli_startup_timeout_retains_failure_and_reaps_child(tmp_path):
    out = tmp_path / "timed-out"
    result = run_cli(out, "--timeout", "0.000001")
    assert result.returncode == 1, result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    evidence = json.loads((out / "evidence.json").read_text())
    assert receipt["status"] == "failed"
    assert evidence["error"]["type"] == "TimeoutError"
    assert evidence["cleanup"]["remaining_children"] == []
    log = [json.loads(line) for line in (out / "probe.log").read_text().splitlines()]
    started = [event["pid"] for event in log if event["event"] == "child_started"]
    assert started
    for pid in started:
        assert not (Path("/proc") / str(pid)).exists()
    assert {entry["pid"] for entry in evidence["cleanup"]["reaped_during_cleanup"]} == set(started)


def test_lifecycle_cli_interruption_retains_partial_receipt_and_reaps_children(tmp_path):
    out = tmp_path / "interrupted"
    with subprocess.Popen(
        [sys.executable, "-I", "-S", "-B", str(SOURCE), "--out", str(out)],
        env={},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            # Wait for retained child ownership, then interrupt before the controls finish.
            deadline = time.monotonic() + 5
            started = []
            while time.monotonic() < deadline:
                log_path = out / "probe.log"
                if log_path.exists():
                    data = log_path.read_bytes()
                    lines = data.split(b"\n")[:-1]
                    started = [json.loads(line)["pid"] for line in lines if json.loads(line)["event"] == "child_started"]
                    if started:
                        break
                select.select([], [], [], 0.001)
            assert started
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 1, (stdout, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["error"]["type"] == "InterruptedError"
    assert evidence["cleanup"]["remaining_children"] == []
    for pid in started:
        assert not (Path("/proc") / str(pid)).exists()
