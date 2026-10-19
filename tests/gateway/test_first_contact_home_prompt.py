"""An operator setup preference is profile-local and does not rewrite delivery homes."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{}, {"gateway": {}}])
async def test_missing_preference_preserves_prompt_default(monkeypatch, config):
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: config)
    monkeypatch.setattr("gateway.run._home_target_env_var", lambda _platform: None)
    runner = SimpleNamespace(config=SimpleNamespace(get_home_channel=lambda _p: None),
                             _deliver_platform_notice=AsyncMock())
    source = SimpleNamespace(platform=Platform.DISCORD, chat_type="thread", profile="default")
    await GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], [])
    runner._deliver_platform_notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiplexed_source_preference_before_runtime_scope(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    member, operator = tmp_path / "member", tmp_path / "operator"
    for home, value in ((member, "false"), (operator, "true")):
        home.mkdir()
        (home / "config.yaml").write_text("gateway:\n  home_channel_prompt: " + value + "\n")
    monkeypatch.setattr("gateway.run._home_target_env_var", lambda _platform: None)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: SimpleNamespace(get_home_channel=lambda _p: None))
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=True, get_home_channel=lambda _p: None),
                             _resolve_profile_home_for_source=lambda source: member if source.profile == "member" else operator,
                             _deliver_platform_notice=AsyncMock())
    # The first-contact phase runs before _run_agent's source-profile scope.
    token = set_hermes_home_override(operator)
    before = {home: (home / "config.yaml").read_bytes() for home in (member, operator)}
    try:
        counts = []
        for profile in ("member", "operator", "member"):
            runner._deliver_platform_notice.reset_mock()
            source = SimpleNamespace(platform=Platform.DISCORD, chat_type="thread", profile=profile)
            await GatewayTurnMixin._hmwa_first_contact_notes(runner, source, [], [])
            counts.append(runner._deliver_platform_notice.await_count)
        assert counts == [0, 1, 0]
        assert before == {home: (home / "config.yaml").read_bytes() for home in (member, operator)}
    finally:
        reset_hermes_home_override(token)


@pytest.mark.asyncio
async def test_real_runner_uses_pinned_routing_identity(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.config import GatewayConfig
    from gateway.session import SessionSource
    from gateway.session_identity import RoutingIdentity
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    member, operator = tmp_path / "member", tmp_path / "operator"
    for home, setting in ((member, "false"), (operator, "true")):
        home.mkdir()
        (home / "config.yaml").write_text("gateway:\n  home_channel_prompt: " + setting + "\n")
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._deliver_platform_notice = AsyncMock()
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread", chat_type="thread", profile="member")
    source._identity = RoutingIdentity("member", "member", member, member)
    monkeypatch.setattr("gateway.run._home_target_env_var", lambda _platform: None)
    token = set_hermes_home_override(operator)
    try:
        assert runner._resolve_profile_home_for_source(source) == member
        await runner._hmwa_first_contact_notes(source, [], [])
        runner._deliver_platform_notice.assert_not_awaited()
    finally:
        reset_hermes_home_override(token)
