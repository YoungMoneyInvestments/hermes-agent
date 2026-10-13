"""--query-file: single-query text arrives verbatim, never shell-interpreted.

Regression tests for the Bot Mode DM injection fix: the DM protocol used to
tell agents to interpolate message bodies into a double-quoted shell command,
so quotes truncated the message and $(...) executed on the sender's machine.
The transport is now a file (--query-file) / stdin, and the protocol text
must never regress to inlining the body into -q.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def _parse(argv):
    sys.path.insert(0, str(REPO))
    try:
        from hermes_cli._parser import build_top_level_parser

        built = build_top_level_parser()
        parser = built[0] if isinstance(built, tuple) else built
        return parser.parse_args(argv)
    finally:
        sys.path.remove(str(REPO))

def test_chat_parser_accepts_query_file():
    args = _parse(["chat", "--query-file", "/tmp/x.txt"])
    assert args.query_file == "/tmp/x.txt"
    assert args.query is None

def test_query_and_query_file_mutually_exclusive(tmp_path):
    """argparse rejects -q + --query-file at parse time (exit 2), no env needed."""
    import pytest

    f = tmp_path / "dm.txt"
    f.write_text("hello", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _parse(["chat", "-q", "x", "--query-file", str(f)])
    assert exc.value.code == 2


def test_invalid_query_file_refused_before_bootstrap(monkeypatch, tmp_path):
    import io
    import pytest
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, "restore_interrupted_pull", lambda: False)
    import hermes_cli.main as entry

    for name in ("_apply_safe_mode", "_apply_user_config_bypass", "_guard_noninteractive_user_config",
                 "_resolve_chat_session_args"):
        monkeypatch.setattr(entry, name, lambda *_args: None)
    monkeypatch.setattr(entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap",
                        lambda **_kwargs: pytest.fail("invalid input reached bootstrap"))
    for query_file, query in (("-", None), (str(tmp_path / "missing.txt"), None), ("-", "conflicting query")):
        monkeypatch.setattr(sys, "stdin", io.StringIO("   "))
        args = _parse(["chat", "--query-file", query_file, "--format", "stream-json"])
        args.query = query
        with pytest.raises(SystemExit) as error:
            entry.cmd_chat(args)
        assert error.value.code == 2


def test_relative_query_file_uses_selected_workspace_before_bootstrap(monkeypatch, tmp_path):
    import pytest
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, "restore_interrupted_pull", lambda: False)
    import hermes_cli.main as entry

    original = tmp_path / "original"
    selected = tmp_path / "selected"
    original.mkdir()
    selected.mkdir()
    (original / "query.txt").write_text("original workspace prompt", encoding="utf-8")
    (selected / "query.txt").write_text("selected workspace prompt", encoding="utf-8")
    monkeypatch.chdir(original)
    for name in ("_apply_safe_mode", "_apply_user_config_bypass", "_guard_noninteractive_user_config"):
        monkeypatch.setattr(entry, name, lambda _args: None)
    monkeypatch.setattr(entry, "_warn_retired_xai_models", lambda: None)
    args = _parse(["chat", "--query-file", "query.txt", "--in", str(selected), "--format", "stream-json"])
    def bootstrap(**_kwargs):
        assert args.query == "selected workspace prompt"
        assert Path.cwd() == selected
        raise RuntimeError("stop before bootstrap side effects")
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", bootstrap)
    with pytest.raises(RuntimeError, match="stop before bootstrap side effects"):
        entry.cmd_chat(args)
