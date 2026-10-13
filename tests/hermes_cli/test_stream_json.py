"""``hermes chat -q … --format stream-json`` emits a parseable JSONL event stream and nothing else on stdout."""

import json
import signal
import sys

import pytest

from hermes_cli.stream_json import StreamJsonEmitter


@pytest.fixture(autouse=True)
def _isolate_checkout_recovery(monkeypatch):
    # These protocol tests never recover a real Git checkout.
    from hermes_cli import _early_recovery
    monkeypatch.setattr(_early_recovery, "restore_interrupted_pull", lambda: False)


def _events(capsys, diagnostics=None):
    captured = capsys.readouterr()
    if diagnostics is not None:
        diagnostics.append(captured.err)
    out = captured.out
    return [json.loads(line) for line in out.splitlines() if line]


def test_emitter_event_stream_is_valid_jsonl(capsys):
    emitter = StreamJsonEmitter(model="test-model", session_id="s-1")
    emitter.on_text_delta("hel")
    emitter.on_text_delta("\n  ")  # whitespace deltas are part of the answer and must be forwarded verbatim
    emitter.on_text_delta("lo")
    emitter.on_text_delta(None)  # the turn-end sentinel the agent sends
    emitter.on_text_delta("")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "x"}, tool_call_id="call-a")
    emitter.on_tool_progress("tool.started", "read_file", "preview", {"path": "y"}, tool_call_id="call-b")
    emitter.on_tool_progress("reasoning.available", "_thinking", "hmm", None)  # not part of the protocol
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-b", result="y")
    emitter.on_tool_progress("tool.completed", "read_file", None, None, tool_call_id="call-a", duration=0.5,
                             is_error=False, result="x" * 6000)
    code = emitter.emit_result({"final_response": "", "failed": True, "error": "boom", "input_tokens": 3}, exit_code=0)

    events = _events(capsys)
    assert [e["type"] for e in events] == ["system", "text", "text", "text", "tool_use", "tool_use", "tool_result",
                                           "tool_result", "result"]
    assert "".join(e["text"] for e in events if e["type"] == "text") == "hel\n  lo"
    assert events[0]["subtype"] == "init" and events[0]["model"] == "test-model"
    assert events[4]["input"] == {"path": "x"} and events[4]["tool_call_id"] == "call-a"
    # concurrent same-name calls: each result pairs with its own start, not the last-started one
    assert [e["tool_call_id"] for e in events if e["type"] == "tool_result"] == ["call-b", "call-a"]
    assert events[6]["duration_ms"] < 500
    assert events[7]["duration_ms"] == 500 and events[7]["output"].endswith("...") and len(events[7]["output"]) == 5003
    assert code == 1 and events[-1] == {**events[-1], "exit_code": 1, "error": "boom", "session_id": "s-1"}
    assert events[-1]["tokens"]["input"] == 3
    assert all("timestamp" in e for e in events)


def _run_stream_json_chat(monkeypatch, capsys, run_conversation, credentials_ok=True, startup_warning=None, diagnostics=None, warning_stage="constructor", worktree_path=None):
    """parser → cmd_chat → cli.main → quiet single-query path with a deterministic fake agent."""
    import cli
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    protocol_stdout = sys.stdout
    exit_callbacks = []
    if startup_warning:
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True)

    class FakeAgent:
        model = "test-model"
        session_id = "session-123"

        def run_conversation(self, **_kwargs):
            if startup_warning and warning_stage == "turn":
                print(startup_warning)
            return run_conversation(self)

    class FakeCLI:
        def __init__(self, **_kwargs):
            assert sys.stdout.isatty() == sys.stderr.isatty()
            if startup_warning and warning_stage == "constructor":
                from rich.console import Console
                from hermes_cli.cli_init_mixin import CLIInitMixin
                self.console = Console()
                CLIInitMixin._init_toolsets(self, ["no_mcp"])
            self.system_prompt = ""
            self.session_id = "session-123"
            self.conversation_history = []
            self.agent = None
            self._active_agent_route_signature = None
            self.tool_progress_mode = None

        def _claim_active_session(self, *_a, **_k):
            return True

        def _ensure_runtime_credentials(self):
            return credentials_ok

        def _resolve_turn_agent_config(self, _query):
            return {"signature": "r", "model": None, "runtime": None, "request_overrides": None}

        def _init_agent(self, **_kwargs):
            self.agent = FakeAgent()
            return True

        def chat(self, _query, images=None):
            print("human output")  # must never reach stdout under stream-json

    from hermes_cli.cli_stream_mixin import CLIStreamMixin
    FakeCLI._console_print = CLIStreamMixin._console_print
    monkeypatch.setattr(cli, "validate_toolset", lambda _name: False)
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_toolset_keys_nowait", lambda: set())
    monkeypatch.setattr(cli, "HermesCLI", FakeCLI)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(cli, "_emit_interrupted_session_end", lambda *_a, **_k: None)
    def worktree_setup(*_args, **_kwargs):
        if startup_warning and warning_stage == "worktree":
            def join():
                print(startup_warning)
                return {"path": "/fixture/worktree", "branch": "fixture", "repo_root": "/fixture/repo"}
            return join
        return None
    if warning_stage == "worktree_cleanup":
        monkeypatch.setattr(cli, "_active_worktree", None)
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
        assert worktree_path is not None
        monkeypatch.setattr(cli, "_setup_worktree", lambda **_k: {"path": str(worktree_path), "branch": "fixture", "repo_root": str(worktree_path.parent)})
        monkeypatch.setattr(cli, "_git_repo_root", lambda: None)
        monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_k: [])
        monkeypatch.setattr(cli, "_worktree_has_unpushed_commits", lambda *_a, **_k: True)
        monkeypatch.setattr(cli, "_repo_is_shallow", lambda _root: False)
        monkeypatch.setitem(cli.CLI_CONFIG, "worktree", True)
    else:
        monkeypatch.setattr(cli, "_start_worktree_setup", worktree_setup)
    if warning_stage == "cleanup":
        monkeypatch.setattr(cli, "_cleanup_done", False)
        monkeypatch.setattr(cli, "_cleanup_in_progress", False)
        monkeypatch.setattr(cli, "_active_agent_ref", None)
        monkeypatch.setattr(cli, "_arm_exit_watchdog", lambda: None)
        monkeypatch.setattr(cli, "_reset_terminal_input_modes_on_exit", lambda: None)
        monkeypatch.setattr(cli, "_should_emit_cleanup_session_finalize", lambda _id: False)
        monkeypatch.setattr(cli, "_shutdown_agent_memory_provider", lambda _agent: None)
        for step, _swallow in cli._CLEANUP_STEPS:
            monkeypatch.setattr(cli, step, lambda: None)
        monkeypatch.setattr(cli, cli._CLEANUP_STEPS[0][0], lambda: cli._cprint(startup_warning))
    else:
        monkeypatch.setattr(cli, "_run_cleanup", lambda: None)
    monkeypatch.setattr(cli.atexit, "register", lambda fn, *a, **k: exit_callbacks.append((fn, a, k)))
    monkeypatch.setattr(signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_entry, "_resolve_use_tui", lambda _args: pytest.fail("TUI resolution consulted"))
    monkeypatch.setattr(cli_entry, "_has_any_provider_configured", lambda: True)
    monkeypatch.setattr(cli_entry, "_start_chat_background_prefetch", lambda: None)
    monkeypatch.setattr(cli_entry, "_pin_kanban_board_env", lambda: None)
    monkeypatch.setattr(cli_entry, "_confirm_startup_expensive_model_override", lambda _a: None)
    monkeypatch.setattr(cli_entry, "_warn_retired_xai_models", lambda: None)
    monkeypatch.setattr("hermes_cli.free_tier_bootstrap.run_bootstrap", lambda **_k: None)
    monkeypatch.setattr("hermes_cli.quiet_single_query.continue_quiet_notify_completions", lambda *_a, **_k: None)

    parser, _, _ = build_top_level_parser()
    args = parser.parse_args(["chat", "-q", "hello", "--format", "stream-json"])
    try:
        with pytest.raises(SystemExit) as exc_info:
            cli_entry.cmd_chat(args)
        for callback, call_args, call_kwargs in reversed(exit_callbacks):
            callback(*call_args, **call_kwargs)
        if startup_warning:
            print(startup_warning)  # An unrelated late callback/thread must also stay off JSONL.
    finally:
        sys.stdout = protocol_stdout  # The standalone process never restores it; this test must.
    return exc_info.value.code, _events(capsys, diagnostics)


def _ok_turn(agent):
    agent.stream_delta_callback("hello")
    agent.tool_progress_callback("tool.started", "read_file", "p", {"path": "f"})
    agent.tool_progress_callback("tool.completed", "read_file", None, None, duration=0.01, result="contents")
    return {"final_response": "hello", "failed": False}


def _interrupted_turn(_agent):
    raise KeyboardInterrupt


@pytest.mark.parametrize("turn, credentials_ok, exit_code, types", [
    (_ok_turn, True, 0, ["system", "text", "tool_use", "tool_result", "result"]),
    (_interrupted_turn, True, 130, ["system", "result"]),
    (_ok_turn, False, 1, ["system", "result"]),  # credentials fail before the agent exists
])
def test_chat_stream_json_implies_quiet_and_closes_with_result(monkeypatch, capsys, turn, credentials_ok, exit_code,
                                                                types):
    """No ``-Q`` needed; stdout is only JSONL; the stream always ends in a ``result`` carrying the exit code."""
    code, events = _run_stream_json_chat(monkeypatch, capsys, turn, credentials_ok=credentials_ok)
    assert code == exit_code
    assert [e["type"] for e in events] == types
    assert events[-1]["exit_code"] == exit_code and events[-1]["session_id"] == "session-123"


@pytest.mark.parametrize("argv, message", [
    (["chat", "--format", "stream-json"], "requires -q/--query"),
    (["chat", "-q", "hi", "--format", "stream-json", "--tui"], "cannot be used with --tui"),
    (["--tui", "chat", "-q", "hi", "--format", "stream-json"], "cannot be used with --tui"),
])
def test_chat_stream_json_rejects_interactive_combinations(monkeypatch, capsys, argv, message):
    import hermes_cli.main as cli_entry
    from hermes_cli._parser import build_top_level_parser

    monkeypatch.setattr(cli_entry, "_launch_tui", lambda *_a, **_k: pytest.fail("TUI launched"))
    parser, _, _ = build_top_level_parser()
    with pytest.raises(SystemExit) as exc_info:
        cli_entry.cmd_chat(parser.parse_args(argv))
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("warning_stage", ["constructor", "worktree", "turn", "cleanup", "worktree_cleanup"])
def test_chat_stream_json_keeps_startup_and_turn_warnings_on_stderr(monkeypatch, capfd, warning_stage, tmp_path):
    from hermes_cli.stream_json import reserve_protocol_stdout
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "stdout", sys.stdout)
        reserve_protocol_stdout()
        with pytest.raises(ValueError, match="explicit protocol stream"):
            StreamJsonEmitter()
    diagnostics = []
    warning = "Warning: Unknown toolsets: no_mcp"
    worktree_path = tmp_path / "kept-worktree"
    worktree_path.mkdir()
    code, events = _run_stream_json_chat(
        monkeypatch, capfd, _ok_turn, startup_warning=warning, diagnostics=diagnostics,
        warning_stage=warning_stage, worktree_path=worktree_path,
    )
    if warning_stage == "worktree_cleanup":
        assert str(worktree_path) in diagnostics[0]
    assert code == 0
    assert warning in diagnostics[0]
    assert events[0]["type"] == "system"
    assert events[-1]["type"] == "result"
    assert sum(event["type"] == "result" for event in events) == 1


def test_protocol_reservation_keeps_original_sink_and_borrows_stderr(monkeypatch):
    import io
    import logging
    from unittest.mock import Mock
    from hermes_cli.stream_json import _DiagnosticStream, reserve_protocol_stdout

    output = io.StringIO()
    diagnostics = Mock(wraps=io.StringIO())
    diagnostics.isatty.return_value = True
    diagnostics.fileno.return_value = 2
    diagnostics.buffer = object()
    logger = logging.Logger("protocol_fixture")
    handler = logging.StreamHandler(output)
    logger.addHandler(handler)
    monkeypatch.setitem(logging.Logger.manager.loggerDict, logger.name, logger)
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "stdout", output)
        scoped.setattr(sys, "stderr", diagnostics)
        assert reserve_protocol_stdout() is output
        assert reserve_protocol_stdout() is output
        assert sys.stdout.isatty() is True
        assert sys.stdout.fileno() == 2
        assert sys.stdout.buffer is diagnostics.buffer
        sys.stdout.close()
        diagnostics.close.assert_not_called()
        logger.warning("captured logger diagnostic")
        assert handler.stream is diagnostics
        emitter = StreamJsonEmitter(output=output)
        emitter.emit_result({"final_response": "fixture"})
    assert [json.loads(line)["type"] for line in output.getvalue().splitlines()] == ["system", "result"]
    ghost = object.__new__(_DiagnosticStream)
    with pytest.raises(AttributeError):
        getattr(ghost, "encoding")
