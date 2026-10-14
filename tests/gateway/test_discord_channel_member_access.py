"""Managed private channels cannot inherit public roles, pairing or bot access."""

import time
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord import adapter as discord_platform
from plugins.platforms.discord.channel_access import channel_member_authorized

GUILD = "111111111111111111"
CHANNEL = "222222222222222222"
MEMBER = "333333333333333333"
OTHER = "444444444444444444"
PUBLIC = "555555555555555555"


def policy():
    return {"channel_member_access": {
        "valid_until": time.time() + 900,
        "channels": {CHANNEL: {"guild_id": GUILD, "users": [MEMBER]}},
    }}


def verdict(extra, *, channel=CHANNEL, guild=GUILD, user=MEMBER):
    return channel_member_authorized(
        extra, channel_ids={channel}, guild_id=guild, user_id=user,
    )


def test_exact_member_and_guild_only():
    extra = policy()
    assert verdict(extra) is True
    assert verdict(extra, user=OTHER) is False
    assert verdict(extra, guild=OTHER) is False
    assert verdict(extra, guild=None) is False
    assert verdict(extra, channel=PUBLIC) is None


@pytest.mark.parametrize("expiry", [None, True, "99999999999", float("nan"), float("inf"), 0, 10**1000])
def test_bad_or_expired_refresh_denies_managed_channel_only(expiry):
    extra = policy()
    extra["channel_member_access"]["valid_until"] = expiry
    assert verdict(extra) is False
    assert verdict(extra, channel=PUBLIC) is None


@pytest.mark.parametrize("grant", [None, {}, {"guild_id": GUILD, "users": "*"},
    {"guild_id": GUILD, "users": [MEMBER, "*"]},
    {"guild_id": GUILD, "users": [MEMBER, True]},
    {"guild_id": int(GUILD), "users": [MEMBER]},
    {"guild_id": GUILD, "users": ["３" * 18]}])
def test_malformed_grant_never_inherits_global_access(grant):
    extra = policy()
    extra["channel_member_access"]["channels"][CHANNEL] = grant
    assert verdict(extra) is False


def test_parent_and_exact_thread_grants_both_must_allow():
    extra = policy()
    assert channel_member_authorized(extra, channel_ids={CHANNEL, PUBLIC}, guild_id=GUILD, user_id=MEMBER)
    extra["channel_member_access"]["channels"][PUBLIC] = {"guild_id": GUILD, "users": [OTHER]}
    assert channel_member_authorized(extra, channel_ids={CHANNEL, PUBLIC}, guild_id=GUILD, user_id=MEMBER) is False


def test_unquoted_managed_channel_key_cannot_fall_through_to_public_access():
    extra = policy()
    extra["channel_member_access"]["channels"] = {int(CHANNEL): {"guild_id": GUILD, "users": [MEMBER]}}
    assert verdict(extra) is False


@pytest.fixture
def adapter(monkeypatch):
    for key in ["DISCORD_ALLOWED_USERS", "DISCORD_ALLOWED_ROLES", "DISCORD_ALLOW_ALL_USERS",
                "DISCORD_ALLOWED_CHANNELS", "DISCORD_IGNORED_CHANNELS", "GATEWAY_ALLOW_ALL_USERS"]:
        monkeypatch.delenv(key, raising=False)
    extra = policy() | {"allowed_channels": [CHANNEL, PUBLIC], "free_response_channels": [CHANNEL, PUBLIC],
                        "no_thread_channels": [CHANNEL, PUBLIC], "allow_bots": "all"}
    a = DiscordAdapter(PlatformConfig(enabled=True, token="test-only", extra=extra))
    a._member_access_policy = SimpleNamespace(extra=lambda: a.config.extra)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999))
    a._allowed_user_ids = {OTHER}
    a._allowed_role_ids = {int(GUILD)}
    a._is_pairing_approved_user = lambda user: True
    a._ready_event.set()
    a._text_batch_delay_seconds = 0
    a.handle_message = AsyncMock()
    return a


def message(*, user=MEMBER, channel=CHANNEL, bot=False, guild=GUILD):
    g = SimpleNamespace(id=int(guild), name="Test guild")
    c = SimpleNamespace(id=int(channel), name="test", guild=g, topic=None)
    return SimpleNamespace(id=int(user), content="Read approved knowledge", mentions=[], attachments=[],
        reference=None, created_at=datetime.now(timezone.utc), channel=c, guild=g,
        author=SimpleNamespace(id=int(user), bot=bot, display_name="Test", name="Test", guild=g, roles=[]),
        type=discord_platform.discord.MessageType.default)


def test_member_guard_overrides_pairing_global_user_role_and_open_access(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    for user, expected in [(MEMBER, True), (OTHER, False)]:
        assert adapter._is_allowed_user(user, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is expected
    assert adapter._is_allowed_user(OTHER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={PUBLIC})
    assert adapter._is_allowed_user(OTHER, is_dm=True)


@pytest.mark.asyncio
async def test_actual_message_dispatch_carries_only_verified_member_grant(adapter):
    assert await adapter._dispatch_discord_message(message())
    event = adapter.handle_message.call_args.args[0]
    assert event.source.user_id == MEMBER
    assert event.source.role_authorized is True
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.pairing_store = SimpleNamespace(is_approved=lambda *a, **k: False)
    assert runner._is_user_authorized(event.source)
    assert runner._is_user_authorized(event.source, allow_adapter_delegation=False) is False
    assert not await adapter._dispatch_discord_message(message(user=OTHER))
    assert adapter.handle_message.await_count == 1


def test_native_yaml_loader_keeps_profile_grants_isolated(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from gateway.config import Platform, load_gateway_config
    homes = [tmp_path / "profile-a", tmp_path / "profile-b"]
    for home, user in zip(homes, [MEMBER, OTHER]):
        home.mkdir()
        extra = policy()
        extra["channel_member_access"]["channels"][CHANNEL]["users"] = [user]
        (home / "config.yaml").write_text(yaml.safe_dump({"discord": extra}))
    for home, expected in [(homes[0], True), (homes[1], False), (homes[0], True)]:
        monkeypatch.setenv("HERMES_HOME", str(home))
        config = load_gateway_config().platforms[Platform.DISCORD]
        assert config.extra["channel_member_access"]["channels"][CHANNEL]["users"] == ([MEMBER] if expected else [OTHER])
        actual = DiscordAdapter(config)
        assert actual._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is expected


def test_bot_bypass_and_wrong_guild_are_denied_before_dispatch(adapter):
    assert adapter._discord_message_admission(message(bot=True), claim=False) == (False, False)
    assert adapter._discord_message_admission(message(user=OTHER, bot=True), claim=False) == (False, False)
    assert adapter._discord_message_admission(message(guild=OTHER), claim=False) == (False, False)


def test_native_slash_auth_and_event_preserve_verified_member_grant(adapter):
    m = message()
    interaction = SimpleNamespace(channel=m.channel, channel_id=m.channel.id, guild=m.guild, guild_id=m.guild.id, user=m.author)
    assert adapter._evaluate_slash_authorization(interaction) == (True, None)
    assert adapter._build_slash_event(interaction, "/status").source.role_authorized is True
    interaction.user = message(user=OTHER).author
    assert adapter._evaluate_slash_authorization(interaction)[0] is False


def test_membership_does_not_grant_dm_access(adapter):
    adapter._is_pairing_approved_user = lambda user: False
    adapter._allowed_role_ids = set()
    assert adapter._is_allowed_user(MEMBER, is_dm=True) is False
    assert adapter._is_allowed_user(OTHER, is_dm=True) is True


@pytest.mark.asyncio
async def test_edited_message_event_uses_same_managed_actor_gate(adapter):
    adapter._platform_events_subscribed = lambda: True
    adapter._platform_event_handler = AsyncMock()
    await adapter._on_platform_message_edit(None, message())
    assert adapter._platform_event_handler.await_count == 1
    source = adapter._platform_event_handler.call_args.args[1]
    assert source.user_id == MEMBER
    assert source.role_authorized is True
    await adapter._on_platform_message_edit(None, message(user=OTHER))
    assert adapter._platform_event_handler.await_count == 1


@pytest.mark.asyncio
async def test_thread_event_inherits_managed_parent_actor_gate(adapter):
    adapter._platform_events_subscribed = lambda: True
    adapter._platform_event_handler = AsyncMock()
    thread = SimpleNamespace(id=int(PUBLIC), parent_id=int(CHANNEL), owner_id=int(MEMBER),
                             guild=SimpleNamespace(id=int(GUILD)), name="Test")
    await adapter._on_platform_thread_create(thread)
    assert adapter._platform_event_handler.await_count == 1
    thread.owner_id = int(OTHER)
    await adapter._on_platform_thread_create(thread)
    assert adapter._platform_event_handler.await_count == 1


@pytest.mark.asyncio
async def test_managed_component_gate_restricts_global_allowlist_and_expires(adapter):
    m = message(user=OTHER)
    interaction = SimpleNamespace(channel=m.channel, channel_id=m.channel.id, guild=m.guild, user=m.author)
    view = discord_platform.SlashConfirmView(session_key="test", confirm_id="test",
        allowed_user_ids={MEMBER, OTHER}, allowed_role_ids=set())
    view._channel_member_gate = adapter._member_interaction_allowed
    assert view._check_auth(interaction) is False
    interaction.user = message().author
    assert view._check_auth(interaction) is True
    adapter.config.extra["channel_member_access"]["valid_until"] = 0
    assert view._check_auth(interaction) is False


@pytest.mark.asyncio
async def test_membership_alone_cannot_approve_component(adapter):
    m = message()
    interaction = SimpleNamespace(channel=m.channel, channel_id=m.channel.id, guild=m.guild, user=m.author)
    view = discord_platform.SlashConfirmView(session_key="test", confirm_id="test",
        allowed_user_ids={OTHER}, allowed_role_ids=set())
    view._channel_member_gate = adapter._member_interaction_allowed
    assert view._check_auth(interaction) is False


@pytest.mark.asyncio
async def test_actual_prompt_send_attaches_member_gate(adapter):
    m = message(user=OTHER)
    interaction = SimpleNamespace(channel=m.channel, channel_id=m.channel.id, guild=m.guild, user=m.author)
    view = discord_platform.SlashConfirmView(session_key="test", confirm_id="test",
        allowed_user_ids={MEMBER, OTHER}, allowed_role_ids=set())
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=123)))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    result = await adapter._send_prompt(CHANNEL, None, lambda c: ({"view": view}, view))
    assert result.success
    assert channel.send.call_args.kwargs["view"]._check_auth(interaction) is False


def test_gateway_managed_gate_precedes_chat_open_and_cached_grants(adapter, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.session import Platform
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.pairing_store = SimpleNamespace(is_approved=lambda *a, **k: True)
    runner._chat_scoped_grant = lambda *a, **k: True
    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    for user, expected in [(MEMBER, True), (OTHER, False)]:
        source = adapter.build_source(chat_id=CHANNEL, chat_type="group", user_id=user,
            guild_id=GUILD, role_authorized=True)
        assert runner._is_user_authorized(source) is expected
    adapter.config.extra["channel_member_access"]["valid_until"] = 0
    source = adapter.build_source(chat_id=CHANNEL, chat_type="group", user_id=MEMBER,
        guild_id=GUILD, role_authorized=True)
    assert runner._is_user_authorized(source) is False


@pytest.mark.asyncio
async def test_dispatch_denies_malformed_managed_parent_even_with_valid_thread_grant(adapter):
    adapter.config.extra["channel_member_access"]["channels"][PUBLIC] = {"guild_id": GUILD, "users": [MEMBER]}
    m = message(channel=PUBLIC)
    m.channel.parent_id = int(CHANNEL)
    assert await adapter._dispatch_discord_message(m)
    m.id += 1
    adapter.config.extra["channel_member_access"]["channels"][CHANNEL] = None
    assert await adapter._dispatch_discord_message(m) is False
    assert adapter.handle_message.await_count == 1


@pytest.mark.asyncio
async def test_real_exec_approval_send_retains_admin_gate_and_member_restriction(adapter):
    adapter._allowed_user_ids = {MEMBER, OTHER}
    adapter._allowed_role_ids = set()
    adapter.config.extra.update(require_admin_for_exec_approval=True, allow_admin_from=[OTHER])
    channel = SimpleNamespace(send=AsyncMock(return_value=SimpleNamespace(id=123)))
    adapter._resolve_channel = AsyncMock(return_value=channel)
    prompt = SimpleNamespace(text="Test approval", choices=["once", "deny", "session", "always"], session_key="test",
        smart_denied=False, chat_id=CHANNEL, metadata=None)
    assert (await adapter._send_exec_approval_prompt(prompt)).success
    view = channel.send.call_args.kwargs["view"]
    for user in [MEMBER, OTHER]:
        m = message(user=user)
        interaction = SimpleNamespace(channel=m.channel, channel_id=m.channel.id, guild=m.guild, user=m.author)
        assert view._check_auth(interaction) is False


def test_snowflakes_fit_unsigned_64_bits():
    extra = policy()
    maximum = str(2**64 - 1)
    extra["channel_member_access"]["channels"][CHANNEL] = {"guild_id": maximum, "users": [maximum]}
    assert verdict(extra, guild=maximum, user=maximum) is True
    extra["channel_member_access"]["channels"][CHANNEL]["users"] = [str(2**64)]
    assert verdict(extra, guild=maximum, user=maximum) is False


def test_live_owning_profile_refresh_routes_prompts_and_tombstones(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from gateway.config import load_gateway_config, Platform
    home = tmp_path / "owner"
    foreign = tmp_path / "foreign"
    home.mkdir()
    foreign.mkdir()
    def write(channels, *, expiry=None):
        (home / "config.yaml").write_text(yaml.safe_dump({"discord": {
            "allowed_channels": [PUBLIC], "free_response_channels": [PUBLIC],
            "channel_member_access": {"valid_until": expiry if expiry is not None else time.time()+900,
                "channels": channels}, "channel_prompts": {CHANNEL: "Member concierge guidance"}}}))
    write({})
    monkeypatch.setenv("HERMES_HOME", str(home))
    a = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    monkeypatch.setenv("HERMES_HOME", str(foreign))
    (foreign / "config.yaml").write_text(yaml.safe_dump({"discord": policy()}))
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is None
    write({CHANNEL: {"guild_id": GUILD, "users": [MEMBER]}})
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is True
    assert {PUBLIC, CHANNEL} <= a._get_allowed_channels()
    assert CHANNEL in a._discord_free_response_channels()
    assert CHANNEL in a._get_no_thread_channels()
    assert a._resolve_channel_prompt(CHANNEL) == "Member concierge guidance"
    assert (foreign / "config.yaml").read_text()
    write({CHANNEL: {"guild_id": GUILD, "users": [OTHER]}})
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is False
    assert a._channel_member_authorized(OTHER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is True
    write({})
    assert a._channel_member_authorized(OTHER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is False
    assert CHANNEL not in a._managed_channel_ids()
    assert PUBLIC in a._get_allowed_channels()
    (home / "config.yaml").rename(home / "config.retained")
    assert a._channel_member_authorized(OTHER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is False


@pytest.mark.parametrize("content", ["discord: [broken", "[]", "discord: {}"])
def test_missing_or_corrupt_live_map_never_revives_known_member(tmp_path, monkeypatch, content):
    import hermes_yaml as yaml
    from gateway.config import load_gateway_config, Platform
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"discord": policy()}))
    a = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    path.write_text(content)
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is False


@pytest.mark.parametrize("failure", ["oversized", "raised"])
def test_native_policy_read_failure_denies_without_crashing(tmp_path, monkeypatch, failure):
    import hermes_yaml as yaml
    from gateway.config import load_gateway_config, Platform
    from hermes_cli import config as native_config
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"discord": policy()}))
    a = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is True
    if failure == "oversized":
        path.write_text(" " * (2 * 1024 * 1024 + 1))
    else:
        def fail():
            raise ValueError("Test unavailable loader")
        monkeypatch.setattr(native_config, "read_raw_config_readonly", fail)
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is False


def test_gateway_actor_callback_failure_is_closed(adapter):
    from gateway.run import GatewayRunner
    from gateway.session import Platform
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    def fail(source):
        raise ValueError("Test policy callback failure")
    adapter.authorize_channel_actor = fail
    source = adapter.build_source(chat_id=CHANNEL, chat_type="group", user_id=MEMBER,
        guild_id=GUILD, role_authorized=True)
    assert runner._is_user_authorized(source) is False


def test_missing_component_channel_context_cannot_skip_member_gate(adapter):
    interaction = SimpleNamespace(channel=None, channel_id=None, guild=SimpleNamespace(id=int(GUILD)),
        user=SimpleNamespace(id=int(OTHER)))
    assert adapter._member_interaction_allowed(interaction) is False


def test_real_gateway_chat_allowlist_is_overridden_only_in_managed_channel(adapter):
    from gateway.run import GatewayRunner
    from gateway.session import Platform
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(platforms={Platform.DISCORD: adapter.config})
    adapter.config.extra["group_allowed_chats"] = [CHANNEL, PUBLIC]
    managed = adapter.build_source(chat_id=CHANNEL, chat_type="group", user_id=OTHER, guild_id=GUILD)
    public = adapter.build_source(chat_id=PUBLIC, chat_type="group", user_id=OTHER, guild_id=GUILD)
    profile = runner._adapter_profile_for_source(managed)
    assert runner._chat_scoped_grant(managed, profile, True, True) is True
    assert runner._is_user_authorized(managed) is False
    assert runner._is_user_authorized(public) is True


def test_real_pairing_file_never_bypasses_managed_membership(adapter, tmp_path, monkeypatch):
    import json
    from gateway.run import GatewayRunner
    from gateway.session import Platform
    from gateway.pairing import PairingStore
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = PairingStore()
    store._approved_path("discord").write_text(json.dumps({OTHER: {"user_name": "Test", "approved_at": time.time()}}))
    assert store.is_approved("discord", OTHER)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner.config = SimpleNamespace(platforms={Platform.DISCORD: adapter.config})
    runner.pairing_store = store
    managed = adapter.build_source(chat_id=CHANNEL, chat_type="group", user_id=OTHER, guild_id=GUILD)
    unmanaged = adapter.build_source(chat_id="999999999999999999", chat_type="group", user_id=OTHER, guild_id=GUILD)
    assert runner._is_user_authorized(managed) is False
    assert runner._is_user_authorized(unmanaged) is True


def test_policy_snapshot_cannot_poison_native_cached_grants(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from plugins.platforms.discord.channel_access import ChannelMemberAccess
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"discord": policy()}))
    reader = ChannelMemberAccess(tmp_path, policy())
    snapshot = reader.extra()
    snapshot["channel_member_access"]["channels"][CHANNEL]["users"].append(OTHER)
    assert verdict(reader.extra(), user=OTHER) is False


def test_managed_prompt_does_not_fall_back_on_expired_route(adapter):
    adapter.config.extra["channel_prompts"] = {CHANNEL: "Old prompt"}
    adapter.config.extra["channel_member_access"]["valid_until"] = 0
    assert adapter._resolve_channel_prompt(CHANNEL) is None


def test_thread_component_resolves_parent_member_grant(adapter):
    thread = SimpleNamespace(id=int(PUBLIC), parent=SimpleNamespace(id=int(CHANNEL)))
    for user, expected in [(MEMBER, True), (OTHER, False)]:
        interaction = SimpleNamespace(channel=thread, channel_id=thread.id,
            guild=SimpleNamespace(id=int(GUILD)), user=SimpleNamespace(id=int(user)))
        assert adapter._member_interaction_allowed(interaction) is expected


@pytest.mark.asyncio
async def test_managed_grant_does_not_create_previously_unset_channel_whitelist(adapter):
    adapter.config.extra["allowed_channels"] = []
    assert await adapter._dispatch_discord_message(message(user=OTHER, channel=PUBLIC))
    assert adapter.handle_message.await_count == 1
    denied = message(user=OTHER)
    denied.id += 1
    assert await adapter._dispatch_discord_message(denied) is False


def test_restart_with_top_level_managed_map_preserves_nested_public_routes(tmp_path, monkeypatch):
    import hermes_yaml as yaml
    from gateway.config import load_gateway_config, Platform
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ["DISCORD_ALLOWED_CHANNELS", "DISCORD_FREE_RESPONSE_CHANNELS", "DISCORD_NO_THREAD_CHANNELS"]:
        monkeypatch.setenv(key, "")
    public = [PUBLIC, OTHER, "666666666666666666", "777777777777777777", "888888888888888888"]
    current = policy()
    current["channel_member_access"]["channels"][PUBLIC] = {"guild_id": GUILD, "users": []}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"platforms": {"discord": {"extra": {
            "allowed_channels": public, "free_response_channels": public, "no_thread_channels": public}}}},
        "discord": current}))
    a = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    assert a.config.extra["allowed_channels"] == public
    assert set(public) <= a._get_allowed_channels()
    assert set(public) <= a._discord_free_response_channels()
    assert set(public) <= a._get_no_thread_channels()
    assert a._channel_member_authorized(MEMBER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={CHANNEL}) is True
    assert a._channel_member_authorized(OTHER, guild=SimpleNamespace(id=int(GUILD)), channel_ids={PUBLIC}) is False
    assert CHANNEL in a._managed_channel_ids()
    assert PUBLIC not in a._managed_channel_ids()
