"""E2E tests for the ``command`` secret source.

These exercise the REAL resolution path: real helper shell scripts written
to a temp dir (chmod +x), real ``/bin/sh -c`` subprocesses, and a real temp
HERMES_HOME with a config.yaml routing ``secrets.provider: command`` through
``hermes_cli.env_loader._apply_external_secret_sources``.

Security invariants under test (ported from the desktop TS provider):

* the requested key travels ONLY via the ``HERMES_SECRET_KEY`` env var —
  never interpolated into the shell string (hostile key names are inert);
* hard timeout + degrade-to-empty on every failure mode, never raise;
* failure logging carries structured fields only — never the command
  string or any secret value.

NOTE: tests assert on key NAMES, lengths, and presence — never log secret
values themselves.
"""

from __future__ import annotations

import os
import stat
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.secret_sources.command import (  # noqa: E402
    CommandSource,
    _run_helper,
    unquote_dotenv_value,
)
from agent.secret_sources.base import (  # noqa: E402
    reset_source_environment,
    set_source_environment,
)
from hermes_cli import env_loader  # noqa: E402


pytestmark = pytest.mark.platforms("posix")  # the command secret provider is POSIX-only


@pytest.mark.parametrize('argv,expected', [
    (['hermes', '--cli', 'chat'], 'terminal-cli'),
    (['hermes', '--cli', 'chat', '--oneshot', '--query-file', 'question.txt'], 'terminal-cli'),
    # Raw profile flags are not proof of terminal mode before main strips them.
    (['hermes', '-p', 'cami', '--cli', 'chat'], 'full'),
    (['hermes', '--profile', 'cami', '--cli', 'chat'], 'full'),
    (['hermes', 'gateway', 'run'], 'full'),
    (['hermes', '--cli', 'gateway', 'run'], 'full'),
    (['hermes', '--cli', 'chat', '--gateway'], 'full'),
    (['hermes', 'chat', '-q', '--cli chat'], 'full'),
    (['hermes'], 'full'),
    (['pytest'], 'full'),
])
def test_helper_context_uses_explicit_runtime_argv_not_inherited_marker(monkeypatch, argv, expected):
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.setenv('HERMES_SECRET_CONTEXT', 'terminal-cli')
    monkeypatch.delenv('_HERMES_GATEWAY', raising=False)
    output = _run_helper('printf "%s" "$HERMES_SECRET_CONTEXT"', '', 1.0, 1024)
    assert output == expected
    # No process-global context is exported, only the helper-child environment.
    assert os.environ['HERMES_SECRET_CONTEXT'] == 'terminal-cli'


def test_gateway_marker_and_multiplex_deny_terminal_context(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['hermes', '--cli', 'chat'])
    monkeypatch.setenv('_HERMES_GATEWAY', '1')
    assert _run_helper('printf "%s" "$HERMES_SECRET_CONTEXT"', '', 1.0, 1024) == 'full'
    monkeypatch.delenv('_HERMES_GATEWAY')
    token = set_source_environment({'HERMES_SECRET_CONTEXT': 'terminal-cli'})
    try:
        assert _run_helper('printf "%s" "$HERMES_SECRET_CONTEXT"', '', 1.0, 1024) == 'full'
    finally:
        reset_source_environment(token)


def test_real_registry_propagates_cli_context_but_scoped_fetches_remain_full(tmp_path, monkeypatch):
    from agent.secret_sources.registry import apply_all
    monkeypatch.setattr(sys, 'argv', ['hermes', '--cli', 'chat', '--oneshot'])
    monkeypatch.delenv('_HERMES_GATEWAY', raising=False)
    helper = _write_helper(tmp_path, 'printf "CONTEXT_TEST_RESULT=%s\\n" "$HERMES_SECRET_CONTEXT"')
    config = {'command': {'enabled': True, 'command': str(helper), 'override_existing': True}}
    for name in ('a', 'b', 'a'):
        home = tmp_path / name
        home.mkdir(exist_ok=True)
        monkeypatch.setenv('HERMES_HOME', str(home))
        apply_all(config, home)
        assert os.environ['CONTEXT_TEST_RESULT'] == 'terminal-cli'
        scoped = {'HERMES_HOME': str(home), 'HERMES_SECRET_CONTEXT': 'terminal-cli'}
        apply_all(config, home, environ=scoped)
        assert scoped['CONTEXT_TEST_RESULT'] == 'full'
        assert os.environ['CONTEXT_TEST_RESULT'] == 'terminal-cli'


@pytest.mark.parametrize('flags', [['-p', 'cami'], ['--profile', 'cami'], ['--profile=cami']])
def test_real_early_profile_strip_precedes_context_detection(monkeypatch, tmp_path, flags):
    # Execute the actual stdlib-only early functions without importing main,
    # which would read live configuration and run credential sources at import.
    import ast
    import re
    from hermes_cli import profiles
    tree = ast.parse((ROOT / 'hermes_cli/main.py').read_text(encoding='utf-8'))
    names = {'_apply_profile_override', '_scan_profile_flag', '_inside_mcp_add_args'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    constant = next(node for node in tree.body if isinstance(node, ast.Assign) and
                    any(isinstance(target, ast.Name) and target.id == '_PROFILE_NAME_RE' for target in node.targets))
    namespace = {'sys': sys, 'os': os, 'Path': Path, 're': re}
    exec(compile(ast.Module(body=[constant, *nodes], type_ignores=[]), '<real-early-profile>', 'exec'), namespace)
    monkeypatch.setattr(profiles, 'resolve_profile_env', lambda name: str(tmp_path / 'profiles' / name))
    monkeypatch.setattr(sys, 'argv', ['hermes', *flags, '--cli', 'chat', '--oneshot'])
    monkeypatch.delenv('_HERMES_GATEWAY', raising=False)
    namespace['_apply_profile_override']()
    assert sys.argv[1:3] == ['--cli', 'chat']
    assert _run_helper('printf "%s" "$HERMES_SECRET_CONTEXT"', '', 1.0, 1024) == 'terminal-cli'


def _write_helper(tmp_path: Path, body: str, name: str = "helper.sh") -> Path:
    """Write a real executable helper script and return its path."""
    script = tmp_path / name
    script.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def test_profile_helper_does_not_inherit_process_secret(monkeypatch):
    monkeypatch.setenv("LEAK_CANARY", "global-secret")
    token = set_source_environment({"PROFILE_ONLY": "profile-value"})
    try:
        output = _run_helper(
            'printf "%s|%s" "${LEAK_CANARY-unset}" "$PROFILE_ONLY"',
            "",
            1.0,
            1024,
        )
    finally:
        reset_source_environment(token)

    assert output == "unset|profile-value"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test starts with a clean source map, applied-home guard, and no
    leftover test keys in os.environ."""
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    for key in ("CMDTEST_API_KEY", "CMDTEST_TOKEN", "CMDTEST_OTHER_KEY"):
        monkeypatch.delenv(key, raising=False)
    yield
    env_loader._SECRET_SOURCES.clear()
    env_loader.reset_secret_source_cache()
    for key in ("CMDTEST_API_KEY", "CMDTEST_TOKEN", "CMDTEST_OTHER_KEY"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Parsing semantics (pure functions, mirroring the TS parseSecretOutput)
# ---------------------------------------------------------------------------


def test_unquote_strips_one_layer_of_matching_quotes():
    assert unquote_dotenv_value('"abc"') == "abc"
    assert unquote_dotenv_value("'abc'") == "abc"
    assert unquote_dotenv_value('""') == ""
    assert unquote_dotenv_value('"') == '"'  # lone quote left intact
    assert unquote_dotenv_value("  plain  ") == "plain"


# ---------------------------------------------------------------------------
# Real-subprocess resolution
# ---------------------------------------------------------------------------


def test_helper_stdout_is_returned_verbatim(tmp_path):
    helper = _write_helper(tmp_path, "printf 'sk-test-bare-12345'")
    value = _run_helper(str(helper), "CMDTEST_API_KEY", 3.0, 1024)
    assert value == "sk-test-bare-12345"


def test_timeout_kills_hung_helper_and_degrades_to_empty(tmp_path):
    helper = _write_helper(tmp_path, "sleep 30")
    start = time.monotonic()
    value = _run_helper(str(helper), "CMDTEST_API_KEY", 2.0, 1024)
    elapsed = time.monotonic() - start
    assert value is None
    assert elapsed < 6.0, f"helper not killed within the bound (took {elapsed:.1f}s)"


def test_failure_logging_never_leaks_command_or_secret(tmp_path, capfd):
    secret_value = "sk-super-secret-value-do-not-log"
    helper = _write_helper(
        tmp_path,
        f"echo '{secret_value}' >&2\nexit 7",
        name="my-distinctive-helper-name.sh",
    )
    value = _run_helper(str(helper), "CMDTEST_API_KEY", 3.0, 1024)
    assert value is None
    captured = capfd.readouterr()
    combined = captured.out + captured.err
    # Structured fields only — never the command string, the helper's
    # stderr, or any secret value.
    assert "my-distinctive-helper-name" not in combined
    assert secret_value not in combined
    assert "code=7" in combined  # the structured field IS logged


def test_fetch_parses_dotenv_blob(tmp_path):
    helper = _write_helper(
        tmp_path,
        "printf 'CMDTEST_API_KEY=sk-applied\\nCMDTEST_TOKEN=tok-applied\\n'",
    )
    result = CommandSource().fetch({"enabled": True, "command": str(helper)}, tmp_path)
    assert result.error is None
    assert result.secrets == {"CMDTEST_API_KEY": "sk-applied", "CMDTEST_TOKEN": "tok-applied"}


# ---------------------------------------------------------------------------
# Dispatch E2E through env_loader against a real temp HERMES_HOME
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    from agent.secret_sources import registry
    registry._reset_registry_for_tests()
    yield
    registry._reset_registry_for_tests()


def test_registry_command_source_applies_and_records_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    helper = _write_helper(
        tmp_path, "printf 'CMDTEST_API_KEY=sk-dispatch\\nCMDTEST_TOKEN=tok-dispatch\\n'"
    )
    (tmp_path / "config.yaml").write_text(
        "secrets:\n"
        "  command:\n"
        "    enabled: true\n"
        f"    command: {helper}\n",
        encoding="utf-8",
    )

    env_loader._apply_external_secret_sources(tmp_path)

    assert os.environ.get("CMDTEST_API_KEY") == "sk-dispatch"
    assert env_loader.get_secret_source("CMDTEST_API_KEY") == "command"
    assert env_loader.get_secret_source("CMDTEST_TOKEN") == "command"
    assert (
        env_loader.format_secret_source_suffix("CMDTEST_API_KEY")
        == " (from Command helper)"
    )


def test_registry_status_line_printed_once_per_home(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    helper = _write_helper(tmp_path, "printf 'CMDTEST_API_KEY=sk-once\\n'")
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n"
        f"    command: {helper}\n",
        encoding="utf-8",
    )

    for _ in range(3):  # idempotency guard: only the first call does work
        env_loader._apply_external_secret_sources(tmp_path)

    err = capsys.readouterr().err
    assert err.count("Command helper: applied 1 secret") == 1


def test_registry_failing_helper_does_not_block_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: exit 9\n",
        encoding="utf-8",
    )
    # Must not raise — config/helper errors never block startup.
    env_loader._apply_external_secret_sources(tmp_path)
    assert env_loader.get_secret_source("CMDTEST_API_KEY") is None


def test_registry_helper_error_prints_remediation(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "secrets:\n  command:\n    enabled: true\n    command: ''\n",
        encoding="utf-8",
    )
    env_loader._apply_external_secret_sources(tmp_path)
    err = capsys.readouterr().err
    assert "secrets.command.command" in err
