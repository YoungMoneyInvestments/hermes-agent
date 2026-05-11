"""Tests for rl_training_tool.py — file handle lifecycle and cleanup.

Verifies that _stop_training_run properly closes log file handles,
terminates processes, and handles edge cases on failure paths.
Inspired by PR #715 (0xbyt4).
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tools.rl_training_tool as rl_training_tool
from tools.rl_training_tool import RunState, _stop_training_run


def _make_run_state(**overrides) -> RunState:
    """Create a minimal RunState for testing."""
    defaults = {
        "run_id": "test-run-001",
        "environment": "test_env",
        "config": {},
    }
    defaults.update(overrides)
    return RunState(**defaults)


class TestStopTrainingRunFileHandles:
    """Verify that _stop_training_run closes log file handles stored as attributes."""

    def test_closes_all_log_file_handles(self):
        state = _make_run_state()
        files = {}
        for attr in ("api_log_file", "trainer_log_file", "env_log_file"):
            fh = MagicMock()
            setattr(state, attr, fh)
            files[attr] = fh

        _stop_training_run(state)

        for attr, fh in files.items():
            fh.close.assert_called_once()
            assert getattr(state, attr) is None

    def test_clears_file_attrs_to_none(self):
        state = _make_run_state()
        state.api_log_file = MagicMock()

        _stop_training_run(state)

        assert state.api_log_file is None

    def test_close_exception_does_not_propagate(self):
        """If a file handle .close() raises, it must not crash."""
        state = _make_run_state()
        bad_fh = MagicMock()
        bad_fh.close.side_effect = OSError("already closed")
        good_fh = MagicMock()
        state.api_log_file = bad_fh
        state.trainer_log_file = good_fh

        _stop_training_run(state)  # should not raise

        bad_fh.close.assert_called_once()
        good_fh.close.assert_called_once()

    def test_handles_missing_file_attrs(self):
        """RunState without log file attrs should not crash."""
        state = _make_run_state()
        # No log file attrs set at all — getattr(..., None) should handle it
        _stop_training_run(state)  # should not raise


class TestStopTrainingRunProcesses:
    """Verify that _stop_training_run terminates processes correctly."""

    def test_terminates_running_processes(self):
        state = _make_run_state()
        for attr in ("api_process", "trainer_process", "env_process"):
            proc = MagicMock()
            proc.poll.return_value = None  # still running
            setattr(state, attr, proc)

        _stop_training_run(state)

        for attr in ("api_process", "trainer_process", "env_process"):
            getattr(state, attr).terminate.assert_called_once()

    def test_does_not_terminate_exited_processes(self):
        state = _make_run_state()
        proc = MagicMock()
        proc.poll.return_value = 0  # already exited
        state.api_process = proc

        _stop_training_run(state)

        proc.terminate.assert_not_called()

    def test_handles_none_processes(self):
        state = _make_run_state()
        # All process attrs are None by default
        _stop_training_run(state)  # should not raise

    def test_handles_mixed_running_and_exited_processes(self):
        state = _make_run_state()
        # api still running
        api = MagicMock()
        api.poll.return_value = None
        state.api_process = api
        # trainer already exited
        trainer = MagicMock()
        trainer.poll.return_value = 0
        state.trainer_process = trainer
        # env is None
        state.env_process = None

        _stop_training_run(state)

        api.terminate.assert_called_once()
        trainer.terminate.assert_not_called()


class TestStopTrainingRunStatus:
    """Verify status transitions in _stop_training_run."""

    def test_sets_status_to_stopped_when_running(self):
        state = _make_run_state(status="running")
        _stop_training_run(state)
        assert state.status == "stopped"

    def test_does_not_change_status_when_failed(self):
        state = _make_run_state(status="failed")
        _stop_training_run(state)
        assert state.status == "failed"

    def test_does_not_change_status_when_pending(self):
        state = _make_run_state(status="pending")
        _stop_training_run(state)
        assert state.status == "pending"

    def test_no_crash_with_no_processes_and_no_files(self):
        state = _make_run_state()
        _stop_training_run(state)  # should not raise
        assert state.status == "pending"


class TestEnvironmentDiscovery:
    """Verify environment discovery refreshes from disk."""

    def test_initialize_environments_rescans_even_when_cache_populated(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "environments"
        env_dir.mkdir()
        monkeypatch.setattr(rl_training_tool, "ENVIRONMENTS_DIR", env_dir)
        rl_training_tool._environments = [
            rl_training_tool.EnvironmentInfo(
                name="stale",
                class_name="StaleEnv",
                file_path="stale.py",
            )
        ]
        (env_dir / "fresh_env.py").write_text(
            "from atroposlib.envs.base import BaseEnv\n"
            "class FreshEnv(BaseEnv):\n"
            "    name = 'fresh'\n",
            encoding="utf-8",
        )

        rl_training_tool._initialize_environments()

        assert [(env.name, env.class_name) for env in rl_training_tool._environments] == [
            ("fresh", "FreshEnv")
        ]


class TestSensitiveRedaction:
    """Verify RL inference logs/streaming do not leak credentials."""

    def test_redacts_exact_and_repr_credentials(self):
        text = (
            "exact sk-or-...alue and "
            "APIServerConfig(api_key='sk-or-v1-...abcd') plus "
            "api_key=\"sk-or-v1-...efgh\" plus "
            "OPENROUTER_API_KEY=sk-or-v1-...ijkl plus "
            "Authorization: Bearer bearer-placeholder-token plus wandb-123"
        )

        redacted = rl_training_tool._redact_sensitive_text(
            text,
            secrets={
                "api_key": "sk-or-v1-secret-value",
                "wandb_api_key": "wandb-123",
            },
        )

        assert "sk-or-...alue" not in redacted
        assert "sk-or-v1-...abcd" not in redacted
        assert "sk-or-v1-...efgh" not in redacted
        assert "sk-or-v1-...ijkl" not in redacted
        assert "bearer-placeholder-token" not in redacted
        assert "wandb-123" not in redacted

        assert "***API_KEY***" in redacted
        assert "***WANDB_API_KEY***" in redacted

    def test_inference_passes_api_key_via_env_not_argv(self, tmp_path, monkeypatch):
        secret = "unit-api-token-value"
        recorded = {}

        class EmptyStream:
            async def readline(self):
                return b""

        class FakeProcess:
            stdout = EmptyStream()
            stderr = EmptyStream()
            returncode = 0

            async def wait(self):
                return 0

            def kill(self):
                self.returncode = -9

        async def fake_create_subprocess_exec(*cmd, **kwargs):
            recorded["cmd"] = cmd
            recorded["env"] = kwargs.get("env", {})
            return FakeProcess()

        monkeypatch.setenv("OPENROUTER_API_KEY", secret)
        monkeypatch.setattr(rl_training_tool, "LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr(rl_training_tool.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
        monkeypatch.setattr(rl_training_tool, "_current_env", "fake_env")
        monkeypatch.setattr(
            rl_training_tool,
            "_environments",
            [
                rl_training_tool.EnvironmentInfo(
                    name="fake_env",
                    class_name="FakeEnv",
                    file_path=str(tmp_path / "fake_env.py"),
                )
            ],
        )

        result = json.loads(asyncio.run(rl_training_tool.rl_test_inference(models=["test/model"])))

        assert result["models_tested"][0]["model"] == "test/model"
        assert secret not in [str(part) for part in recorded["cmd"]]
        assert recorded["env"]["ATROPOS_OPENAI_API_KEY"] == secret
        assert recorded["env"]["ATROPOS_ALLOW_DUMMY_MANAGED_SERVER"] == "1"


class TestMatrixLSTMEnvironment:
    """Verify Matrix-LSTM env is portable and configurable."""

    def _load_matrix_module(self, monkeypatch, repo_path: Path):
        monkeypatch.setenv("MATRIX_LSTM_REPO_PATH", str(repo_path))
        module_path = (
            Path(__file__).resolve().parents[2]
            / "tinker-atropos"
            / "tinker_atropos"
            / "environments"
            / "matrix_lstm_tinker.py"
        )
        spec = importlib.util.spec_from_file_location("matrix_lstm_tinker_test", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["matrix_lstm_tinker_test"] = module
        spec.loader.exec_module(module)
        return module

    def test_repo_path_comes_from_environment_variable(self, tmp_path, monkeypatch):
        module = self._load_matrix_module(monkeypatch, tmp_path)

        assert module.get_matrix_lstm_repo() == tmp_path
        assert module.get_data_dir() == tmp_path / "data" / "inefficiency_gate"

    def test_source_does_not_hardcode_user_home_path(self):
        module_path = (
            Path(__file__).resolve().parents[2]
            / "tinker-atropos"
            / "tinker_atropos"
            / "environments"
            / "matrix_lstm_tinker.py"
        )
        source = module_path.read_text(encoding="utf-8")

        assert "/Users/cameronbennion" not in source
