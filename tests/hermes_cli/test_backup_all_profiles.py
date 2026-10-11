"""Tests for the #66140 fix: pre-update snapshots cover every profile."""

import json
import re
from pathlib import Path

import pytest

import hermes_cli.backup as backup
import hermes_cli.update_cmd_maint as update_maint


def _mk_profile(home: Path, jobs: int = 0) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    if jobs:
        cron = home / "cron"
        cron.mkdir(exist_ok=True)
        payload = {"jobs": [{"id": f"j{i}"} for i in range(jobs)]}
        (cron / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")
    return home


@pytest.fixture()
def profiles(monkeypatch, tmp_path):
    """default (invoking) + work + sparks profile homes."""
    default_home = _mk_profile(tmp_path / "home", jobs=3)
    work = _mk_profile(tmp_path / "home" / "profiles" / "work", jobs=5)
    sparks = _mk_profile(tmp_path / "home" / "profiles" / "sparks", jobs=0)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: tmp_path / "home" / "profiles"
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._PROFILE_ID_RE",
        re.compile(r"^[a-z0-9][a-z0-9_-]*$"),
        raising=False,
    )
    return {"default": default_home, "work": work, "sparks": sparks}


class TestSiblingEnumeration:
    def test_excludes_invoking_profile(self, profiles):
        names = [n for n, _ in backup._sibling_profile_homes(profiles["default"])]
        assert names == ["sparks", "work"]

    def test_invoked_from_named_profile_includes_default(self, profiles):
        names = [n for n, _ in backup._sibling_profile_homes(profiles["work"])]
        assert names == ["default", "sparks"]

    def test_never_raises(self, monkeypatch, tmp_path):
        def _boom():
            raise RuntimeError("no profiles module")

        monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", _boom)
        assert backup._sibling_profile_homes(tmp_path) == []


class TestAllProfileSnapshots:
    def test_config_only_active_home_skips_automatic_backup(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "config.yaml").write_text(
            "profile:\n  config_only: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        calls = []
        monkeypatch.setattr(
            update_maint, "_resolve_pre_update_backup_mode", lambda args: calls.append("mode") or "quick"
        )
        monkeypatch.setattr(update_maint, "_run_quick_snapshots", lambda: calls.append("snapshots") or None)
        monkeypatch.setattr(update_maint, "_run_full_backup", lambda: calls.append("full"))

        assert update_maint._run_pre_update_backup(object()) is None
        assert calls == ["mode", "snapshots"]
        output = capsys.readouterr().out
        assert "skipped (config-only)" in output
        assert "snapshot FAILED" not in output
        assert not (tmp_path / "state-snapshots").exists()
        assert not (tmp_path / "backups").exists()

    def test_quick_snapshot_skips_config_only_active_but_keeps_sibling_recovery(
        self, profiles, monkeypatch
    ):
        (profiles["default"] / "config.yaml").write_text(
            "profile:\n  config_only: true\n", encoding="utf-8"
        )
        monkeypatch.setenv("HERMES_HOME", str(profiles["default"]))
        calls = []

        def fake_snapshot(*, label, hermes_home=None, keep=None, max_file_size=None):
            calls.append((label, hermes_home))
            return f"snapshot-{Path(hermes_home).name}"

        monkeypatch.setattr(backup, "create_quick_snapshot", fake_snapshot)
        assert update_maint._run_quick_snapshots() is None

        assert calls == [
            ("pre-update", profiles["sparks"]),
            ("pre-update", profiles["work"]),
        ]
        from hermes_cli import update_cmd_config

        assert update_cmd_config._LAST_SIBLING_SNAPSHOTS == {
            "sparks": "snapshot-sparks",
            "work": "snapshot-work",
        }

    def test_config_only_sibling_is_skipped_without_blocking_healthy_sibling(
        self, profiles, monkeypatch
    ):
        (profiles["sparks"] / "config.yaml").write_text(
            "profile:\n  config_only: true\n", encoding="utf-8"
        )
        calls = []

        def fake_snapshot(*, label, hermes_home, keep=None, max_file_size=None):
            calls.append((label, hermes_home))
            return "snapshot-id"

        monkeypatch.setattr(backup, "create_quick_snapshot", fake_snapshot)
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"]
        )

        assert result == {"work": "snapshot-id"}
        assert calls == [("pre-update", profiles["work"])]

    def test_each_sibling_snapshotted_into_own_home(self, profiles):
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        assert set(result) == {"work", "sparks"}
        for name, snap_id in result.items():
            snap_dir = profiles[name] / "state-snapshots" / snap_id
            assert snap_dir.is_dir()
            assert (snap_dir / "config.yaml").is_file()
            assert "pre-update" in snap_id
        # invoking profile untouched by THIS call
        assert not (profiles["default"] / "state-snapshots").exists()

    def test_size_cap_forwarded(self, profiles):
        big = profiles["work"] / "state.db"
        big.write_bytes(b"\x00" * 4096)
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1, max_file_size=1024
        )
        snap_dir = profiles["work"] / "state-snapshots" / result["work"]
        assert not (snap_dir / "state.db").exists()  # capped out
        assert (snap_dir / "config.yaml").is_file()  # small files captured

    def test_one_failing_sibling_does_not_block_others(self, profiles, monkeypatch):
        real = backup.create_quick_snapshot

        def _flaky(label=None, hermes_home=None, keep=None, max_file_size=None):
            if hermes_home == profiles["work"]:
                raise OSError("disk full")
            return real(
                label=label, hermes_home=hermes_home, keep=keep,
                max_file_size=max_file_size,
            )

        monkeypatch.setattr(backup, "create_quick_snapshot", _flaky)
        result = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"]
        )
        assert "sparks" in result and "work" not in result


class TestPerProfileCronRestore:
    def test_lost_jobs_restored_from_own_snapshot(self, profiles):
        snaps = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        # simulate the migration emptying work's jobs.json
        jobs_path = profiles["work"] / "cron" / "jobs.json"
        jobs_path.write_text(json.dumps({"jobs": []}), encoding="utf-8")

        restored = backup.restore_cron_jobs_all_profiles(
            snaps, invoking_home=profiles["default"]
        )
        assert len(restored) == 1
        assert restored[0]["profile"] == "work"
        assert restored[0]["job_count"] == 5
        live = json.loads(jobs_path.read_text(encoding="utf-8"))
        assert len(live["jobs"]) == 5

    def test_healthy_profiles_untouched(self, profiles):
        snaps = backup.create_pre_update_snapshots_all_profiles(
            invoking_home=profiles["default"], keep=1
        )
        assert backup.restore_cron_jobs_all_profiles(
            snaps, invoking_home=profiles["default"]
        ) == []

    def test_empty_map_is_noop(self, profiles):
        assert backup.restore_cron_jobs_all_profiles({}) == []
