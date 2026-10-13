"""Expiring, exact-member grants; configured Discord snowflakes are quoted strings."""

import math
import time
import logging
import copy
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ChannelMemberAccess:
    """Read only the owning profile; retain revoked/unreadable routes as deny tombstones."""

    def __init__(self, home: Path, initial: dict):
        self.home = home
        self.known_channels = set()
        self._remember(initial)

    def _remember(self, extra: dict) -> None:
        policy = extra.get("channel_member_access")
        channels = policy.get("channels") if isinstance(policy, dict) else None
        if isinstance(channels, dict):
            self.known_channels.update(str(key) for key in channels if _snowflake(str(key)))

    def extra(self) -> dict:
        """Owned managed-policy snapshot; never expose native cache objects to callers."""
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config import read_raw_config_readonly, FailedConfigRead
        from gateway.config_loader import platform_section

        raw = None
        try:
            if (self.home / "config.yaml").stat().st_size <= 2 * 1024 * 1024:
                token = set_hermes_home_override(self.home)
                try:
                    loaded = read_raw_config_readonly()
                    if not isinstance(loaded, FailedConfigRead):
                        raw = loaded
                finally:
                    reset_hermes_home_override(token)
        except Exception as exc:
            logger.debug("Managed channel policy unavailable (%s)", type(exc).__name__)
        section = {}
        if isinstance(raw, dict):
            gateway = raw.get("gateway", {})
            nested = gateway.get("platforms", {}) if isinstance(gateway, dict) else {}
            section, _ = platform_section(raw, "discord", nested)
            section = section if isinstance(section, dict) else {}
        extra = {**(section.get("extra") if isinstance(section.get("extra"), dict) else {}), **section}
        self._remember(extra)
        policy = extra.get("channel_member_access")
        if isinstance(policy, dict) and isinstance(policy.get("channels"), dict):
            channels = {key: None for key in self.known_channels}
            channels.update(copy.deepcopy(policy["channels"]))
            policy = {**policy, "channels": channels}
        elif "channel_member_access" not in extra:
            if not self.known_channels:
                return {}
            policy = {"valid_until": 0, "channels": {key: None for key in self.known_channels}}
        return {"channel_member_access": policy,
                "channel_prompts": copy.deepcopy(extra.get("channel_prompts", {}))}


def active_channel_ids(extra: dict) -> set[str]:
    policy = extra.get("channel_member_access")
    channels = policy.get("channels") if isinstance(policy, dict) else None
    if not isinstance(channels, dict):
        return set()
    return {key for key, grant in channels.items()
            if _snowflake(key) and isinstance(grant, dict)
            and isinstance(grant.get("users"), list) and grant["users"]
            and channel_member_authorized(extra, channel_ids={key}, guild_id=grant.get("guild_id"),
                                         user_id=grant["users"][0]) is True}


def channel_member_authorized(
    extra: Any, *, channel_ids: set[str] | None, guild_id: Any, user_id: str,
) -> bool | None:
    """Operator-owned private route map; every matched channel/parent must grant access.

    None outside managed channels. Membership replaces the global admission allowlist
    only in that exact guild/channel, and never grants an action or DM permission.
    """
    if not channel_ids or not isinstance(extra, dict) or "channel_member_access" not in extra:
        return None
    policy = extra["channel_member_access"]
    if not isinstance(policy, dict) or not isinstance(policy.get("channels"), dict):
        return False
    channels = policy["channels"]
    if any(not _snowflake(key) for key in channels):
        return False
    matched = [channels[key] for key in channel_ids if key in channels]
    if not matched:
        return None
    valid_until = policy.get("valid_until")
    if isinstance(valid_until, bool) or not isinstance(valid_until, (int, float)):
        return False
    try:
        if not math.isfinite(valid_until) or valid_until <= time.time():
            return False
    except OverflowError:
        return False
    guild = str(guild_id or "")
    user = str(user_id or "")
    if not _snowflake(guild) or not _snowflake(user):
        return False
    for grant in matched:
        if not isinstance(grant, dict) or grant.get("guild_id") != guild:
            return False
        users = grant.get("users")
        if not isinstance(users, list) or not users or not all(_snowflake(uid) for uid in users):
            return False
        if user not in users:
            return False
    return True


def _snowflake(value: Any) -> bool:
    return (isinstance(value, str) and value.isascii() and value.isdigit()
            and 1 <= len(value) <= 20 and value[0] != "0" and int(value) <= 2**64 - 1)
