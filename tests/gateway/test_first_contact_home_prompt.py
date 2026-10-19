"""Disabling operator onboarding must not change reply routing or hide failures."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,expected", [(False, 0), ("false", 0), (0, 0), (True, 1), (None, 1)])
async def test_home_prompt_opt_out_preserves_default(monkeypatch, setting, expected):
    config = {"gateway": {"home_channel_prompt": setting}}
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._home_target_env_var", lambda _platform: None)
    runner = SimpleNamespace(config=SimpleNamespace(get_home_channel=lambda _p: None),
                             _deliver_platform_notice=AsyncMock())
    source = SimpleNamespace(platform=Platform.DISCORD, chat_type="thread", profile="default")
    notes = []
    await GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], notes)
    assert runner._deliver_platform_notice.await_count == expected
    assert notes == []


@pytest.mark.asyncio
async def test_existing_history_never_repeats_home_prompt(monkeypatch):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    runner = SimpleNamespace(_deliver_platform_notice=AsyncMock())
    await GatewayTurnMixin._hmwa_first_contact_notes(runner, object(), [{"role": "user"}], [])
    runner._deliver_platform_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_config_loader_is_profile_scoped_and_read_only(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from gateway.run import _load_gateway_config
    monkeypatch.setattr("gateway.run._home_target_env_var", lambda _platform: None)
    homes = [tmp_path / "member", tmp_path / "operator"]
    for home, value in zip(homes, ("false", "true")):
        home.mkdir()
        (home / "config.yaml").write_text("gateway:\n  home_channel_prompt: " + value + "\n")
    before = {str(h): (h / "config.yaml").read_bytes() for h in homes}
    # A -> B -> A uses the same context-local home binding as the multiplexed turn wrapper.
    counts = []
    for home in (homes[0], homes[1], homes[0]):
        token = set_hermes_home_override(home)
        try:
            assert _load_gateway_config()["gateway"]["home_channel_prompt"] is (home == homes[1])
            runner = SimpleNamespace(config=SimpleNamespace(get_home_channel=lambda _p: None),
                                     _deliver_platform_notice=AsyncMock())
            source = SimpleNamespace(platform=Platform.DISCORD, chat_type="thread", profile="default")
            await GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], [])
            counts.append(runner._deliver_platform_notice.await_count)
        finally:
            reset_hermes_home_override(token)
    assert counts == [0, 1, 0]
    assert before == {str(h): (h / "config.yaml").read_bytes() for h in homes}
