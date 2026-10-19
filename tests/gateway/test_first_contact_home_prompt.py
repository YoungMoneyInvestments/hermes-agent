"""Disabling operator onboarding must not change reply routing or hide failures."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run_turn import GatewayTurnMixin


@pytest.mark.asyncio
@pytest.mark.parametrize("setting,expected", [(False, 0), (True, 1), (None, 1)])
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
