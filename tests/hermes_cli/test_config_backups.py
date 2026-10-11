"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
from pathlib import Path

import pytest

from hermes_cli.config_backups import backup_config, list_config_backups


def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(3):
        cfg.write_text(f"model: {i}\n")
        backup_config(cfg, "pre-setup", keep=2)
    kept = list_config_backups(cfg, "pre-setup")
    assert len(kept) == 2 and kept[0].read_text() == "model: 2\n"
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    assert (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))


@pytest.mark.parametrize(
    "contents", ["profile:\n  config_only: true\n"]
)
def test_config_only_home_skips_backup_before_directory_creation(tmp_path, contents):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(contents, encoding="utf-8")

    assert backup_config(cfg, "good") is None
    assert not (tmp_path / "backups").exists()
    assert cfg.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("contents", ["model: a\n", "profile: null\n", "profile:\n  config_only: false\n"])
def test_legacy_home_still_reaches_backup_writer_with_absent_null_or_false_profile(
    tmp_path, monkeypatch, contents
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(contents, encoding="utf-8")
    calls = []

    def record_copy(source, destination):
        calls.append((Path(source), Path(destination)))
        return Path(destination)

    monkeypatch.setattr("hermes_cli.config_backups.shutil.copy2", record_copy)

    backup = backup_config(cfg, "good")

    assert backup is not None
    assert backup.parent == tmp_path / "backups" / "config"
    assert calls == [(cfg, backup)]


def test_unreadable_profile_config_keeps_corrupt_backup_recovery(tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    contents = "profile:\n  config_only: 'true'\n"
    cfg.write_text(contents, encoding="utf-8")
    real_read_text = Path.read_text

    def denied_read(path, *args, **kwargs):
        if path == cfg:
            raise PermissionError("read denied for test")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied_read)

    backup = backup_config(cfg, "corrupt")

    assert backup is not None and backup.read_text(encoding="utf-8") == contents


def test_malformed_yaml_keeps_corrupt_backup_recovery(tmp_path):
    cfg = tmp_path / "config.yaml"
    contents = "profile: [broken\n"
    cfg.write_text(contents, encoding="utf-8")

    backup = backup_config(cfg, "corrupt")

    assert backup is not None and backup.read_text(encoding="utf-8") == contents
