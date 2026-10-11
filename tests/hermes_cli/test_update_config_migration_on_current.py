"""Migration UX policy; selected-child and real profile writes live in completion tests."""
import sys

import pytest

from hermes_cli import config, update_cmd


@pytest.mark.parametrize('case,expected', [
    ('format', [(False, True)]), ('warnings', [(False, True)]),
    ('migration-error', [(False, True)]), ('current', []), ('ahead', []), ('read-error', []),
    ('yes', [(False, False)]), ('tty-yes', [(True, False)]), ('tty-decline', []),
    ('noninteractive', [(False, False)]), ('gateway', [(False, False)]), ('eof', []), ('unicode', []),
])
def test_migration_policy(monkeypatch, capsys, case, expected):
    prompts, calls = [], []
    named = case in {'yes', 'tty-yes', 'tty-decline', 'noninteractive', 'gateway', 'eof', 'unicode'}
    monkeypatch.setattr(config, 'get_missing_env_vars', lambda **_: [
        {'name': 'NEW_TOKEN', 'description': 'new credential'}] if named else [])
    monkeypatch.setattr(config, 'get_missing_config_fields', lambda: [
        {'key': 'new.option', 'description': 'new option'}] if named else [])

    def version(**kwargs):
        assert kwargs == {'raise_on_parse_error': True}
        if case == 'read-error':
            raise RuntimeError('cannot read config')
        return (4 if case == 'ahead' else 3 if case == 'current' else 2, 3)

    def migrate(*, interactive, quiet):
        calls.append((interactive, quiet))
        if case == 'migration-error':
            raise RuntimeError('cannot write config')
        return {'env_added': [], 'config_added': ['setting reset'] if case == 'warnings' else [],
                'warnings': ['personality reset'] if case == 'warnings' else []}

    def prompt(text):
        prompts.append(text)
        if case == 'eof':
            raise EOFError()
        if case == 'unicode':
            raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid')
        return 'n' if case == 'tty-decline' else 'y'

    monkeypatch.setattr(config, 'check_config_version', version)
    monkeypatch.setattr(config, 'migrate_config', migrate)
    monkeypatch.setattr(update_cmd, '_migrate_sibling_profile_configs', lambda: [])
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr(sys.stdout, 'isatty', lambda: case != 'noninteractive')
    monkeypatch.setattr('builtins.input', prompt)
    gateway_prompts = []
    monkeypatch.setattr(update_cmd, '_gateway_prompt', lambda text, default: gateway_prompts.append((text, default)) or 'y')
    update_cmd._check_and_apply_config_migration(assume_yes=case == 'yes', gateway_mode=case == 'gateway')
    assert calls == expected
    assert bool(prompts) is (case in {'tty-yes', 'tty-decline', 'eof', 'unicode'})
    assert bool(gateway_prompts) is (case == 'gateway')
    output = capsys.readouterr().out
    if named:
        assert 'NEW_TOKEN' in output and 'new.option' in output
        assert 'new credential' in output and 'new option' in output
        if case in {'yes', 'noninteractive', 'gateway'}:
            assert 'API keys require manual entry' in output
        elif not expected:
            assert 'hermes config migrate' in output
    elif case in {'current', 'ahead'}:
        assert 'Configuration is up to date' in output
    elif case == 'read-error':
        assert 'Could not check config version' in output
    elif case == 'migration-error':
        assert 'Config format update failed: cannot write config' in output
    else:
        assert 'v2 → v3' in output and 'no new settings to configure' in output
        if case == 'warnings':
            assert 'setting reset' in output and 'personality reset' in output


def test_config_only_update_skips_active_migration_but_keeps_sibling_pass(tmp_path, monkeypatch, capsys):
    import hermes_constants

    active = tmp_path / "profiles" / "staging"
    active.mkdir(parents=True)
    launch_home = tmp_path / "launch-home"
    launch_home.mkdir()
    config_path = active / "config.yaml"
    config_path.write_text(
        "_config_version: 1\nprofile:\n  config_only: true\n", encoding="utf-8"
    )
    before = config_path.read_bytes()
    sibling_passes = []
    active_calls = []

    monkeypatch.setattr(hermes_constants, "get_process_hermes_home", lambda: launch_home)
    monkeypatch.setattr(config, "get_missing_env_vars", lambda **kwargs: active_calls.append("env") or [])
    monkeypatch.setattr(config, "get_missing_config_fields", lambda: active_calls.append("fields") or [])
    monkeypatch.setattr(config, "check_config_version", lambda **kwargs: active_calls.append("version") or (1, 2))
    monkeypatch.setattr(
        config, "migrate_config",
        lambda **kwargs: active_calls.append("migrate") or {"env_added": [], "config_added": []},
    )
    monkeypatch.setattr(
        update_cmd, "_migrate_sibling_profile_configs", lambda: sibling_passes.append(True) or []
    )

    token = hermes_constants.set_hermes_home_override(active)
    try:
        update_cmd._check_and_apply_config_migration()
    finally:
        hermes_constants.reset_hermes_home_override(token)

    assert sibling_passes == [True]
    assert active_calls == []
    assert config_path.read_bytes() == before
    assert "Automatic config migration skipped" in capsys.readouterr().out


@pytest.mark.parametrize('named', [False, True])
def test_update_copies_bundled_skill_bytes_to_default_active_and_sibling(tmp_path, monkeypatch, named):
    from pathlib import Path
    from hermes_cli import update_cmd_maint

    home = tmp_path / '.hermes'
    homes = [home, home / 'profiles/active', home / 'profiles/sibling'] if named else [home]
    for profile in homes:
        profile.mkdir(parents=True, exist_ok=True)
        (profile / 'config.yaml').write_text('{}\n', encoding='utf-8')
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(homes[1] if named else home))
    monkeypatch.setattr('plugins.memory.honcho.cli.sync_honcho_profiles_quiet', lambda: [])
    update_cmd_maint._sync_profiles_after_update()
    bundled = Path(__file__).resolve().parents[2] / 'skills'
    witness = next(bundled.rglob('SKILL.md'))
    for profile in homes:
        assert (profile / 'skills' / witness.relative_to(bundled)).read_bytes() == witness.read_bytes()
