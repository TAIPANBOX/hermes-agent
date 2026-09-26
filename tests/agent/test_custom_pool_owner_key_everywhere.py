"""A bare-``custom`` runtime is bound to the pool its OWN key can own, never to a same-URL sibling's.

Matching a custom pool by ``base_url`` alone names the pool of the FIRST entry on the URL, which
may be a named sibling holding a different key. The main model's resolver already skips such a
pool (``owner_api_key``); these are the other places that still matched by URL alone:

* restore after a cross-provider fallback: ``resolve_runtime_pool_key`` loaded the sibling's pool
  and ``_rebind_primary_credential_pool`` swapped the sibling's key into the primary runtime;
* ``_finalize_routing`` and the 429/401 pool-mutation guard rejected the model's OWN pool, so it
  was dropped at init and never rotated;
* ``_resolve_direct_alias_runtime`` (bare ``custom`` + explicit base_url) resolved from the
  sibling's pool instead of using the explicit key;
* a bare-``custom`` delegated child leased the sibling's pool.

With no key of its own a runtime has nothing to tell entries apart by, and keeps the URL-only
lookup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml
import pytest

URL = "https://llm.example.test/v1"


def _write_home(tmp_path, monkeypatch, config, *, pools=None):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENROUTER_API_KEY", "CUSTOM_BASE_URL", "OPENROUTER_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    (home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}, "credential_pool": pools or {}}))
    return home


def _model(api_key=None):
    model = {"default": "m1", "provider": "custom", "base_url": URL, "api_mode": "chat_completions"}
    if api_key is not None:
        model["api_key"] = api_key
    return model


def _sibling_then_own(tmp_path, monkeypatch, *, model_key="main-key"):
    """A named sibling with its own literal key listed FIRST on the URL, then the model's own
    credential-less entry (#100413), which carries the ``model_config`` row."""
    _write_home(tmp_path, monkeypatch, {
        "model": _model(model_key),
        "custom_providers": [
            {"name": "Second", "base_url": URL, "api_key": "second-key", "model": "m1"},
            {"name": "Own", "base_url": URL, "model": "m1"},
        ],
    })


def _only_sibling(tmp_path, monkeypatch, *, model_key="main-key", pools=None):
    _write_home(tmp_path, monkeypatch, {
        "model": _model(model_key),
        "custom_providers": [{"name": "Second", "base_url": URL, "api_key": "second-key", "model": "m1"}],
    }, pools=pools)


def _selected_key(pool_key):
    from agent.credential_pool import load_pool

    entry = load_pool(pool_key).select()
    return entry.access_token if entry is not None else None


# ── (A) credential_pool: resolve_runtime_pool_key / credential_pool_matches_provider ─────────


def test_runtime_pool_key_prefers_the_pool_the_owner_key_can_own(tmp_path, monkeypatch):
    from agent.credential_pool import resolve_runtime_pool_key

    _sibling_then_own(tmp_path, monkeypatch)

    key = resolve_runtime_pool_key("custom", URL, owner_api_key="main-key")

    assert key == "custom:own"
    assert _selected_key(key) == "main-key"


def test_runtime_pool_key_never_names_a_sibling_the_owner_key_cannot_own(tmp_path, monkeypatch):
    """Only a sibling on the URL: the runtime has no pool of its own and fails closed to its
    plain identity rather than borrowing the sibling's."""
    from agent.credential_pool import resolve_runtime_pool_key

    _only_sibling(tmp_path, monkeypatch)

    assert resolve_runtime_pool_key("custom", URL, owner_api_key="main-key") == "custom"


def test_runtime_pool_key_without_an_owner_key_stays_url_only(tmp_path, monkeypatch):
    from agent.credential_pool import resolve_runtime_pool_key

    _sibling_then_own(tmp_path, monkeypatch, model_key=None)

    assert resolve_runtime_pool_key("custom", URL) == "custom:second"
    assert resolve_runtime_pool_key("custom", URL, owner_api_key=None) == "custom:second"
    assert resolve_runtime_pool_key("custom", URL, owner_api_key="") == "custom:second"


def test_pool_match_with_owner_key_accepts_own_pool_and_rejects_sibling(tmp_path, monkeypatch):
    from agent.credential_pool import credential_pool_matches_provider, load_pool

    _sibling_then_own(tmp_path, monkeypatch)
    own, sibling = load_pool("custom:own"), load_pool("custom:second")

    assert credential_pool_matches_provider(own, "custom", base_url=URL, owner_api_key="main-key")
    assert not credential_pool_matches_provider(sibling, "custom", base_url=URL, owner_api_key="main-key")
    # URL-only verdict unchanged when no key is known.
    assert not credential_pool_matches_provider(own, "custom", base_url=URL)
    assert credential_pool_matches_provider(sibling, "custom", base_url=URL)


def test_pool_match_accepts_a_pool_whose_row_is_the_runtime_key(tmp_path, monkeypatch):
    """A keyless model resolved by URL alone is bound to one of that entry's rows (here a
    ``hermes auth add`` row that differs from the entry's literal key): that pool stays its own."""
    from agent.credential_pool import credential_pool_matches_provider, load_pool

    manual = {"id": "m1", "source": "manual", "auth_type": "api_key", "access_token": "manual-key",
              "base_url": URL, "label": "manual", "priority": 0}
    _only_sibling(tmp_path, monkeypatch, model_key=None, pools={"custom:second": [manual]})
    pool = load_pool("custom:second")

    assert credential_pool_matches_provider(pool, "custom", base_url=URL, owner_api_key="manual-key")
    assert not credential_pool_matches_provider(pool, "custom", base_url=URL, owner_api_key="other-key")


def test_finalize_routing_keeps_the_models_own_pool(tmp_path, monkeypatch):
    from agent.agent_init import _finalize_routing
    from agent.credential_pool import load_pool

    _sibling_then_own(tmp_path, monkeypatch)
    monkeypatch.setattr("hermes_cli.anon_auth.pin_model_for_route", lambda provider, base_url, model: model)
    pool = load_pool("custom:own")
    agent = SimpleNamespace(
        provider="custom", base_url=URL, model="m1", api_mode="chat_completions", _credential_pool=pool,
        _get_transport=lambda: None, _is_azure_openai_url=lambda: False, _is_openrouter_url=lambda: False,
        _is_direct_openai_url=lambda: False,
        _provider_model_requires_responses_api=lambda model, provider=None: False, _transport_cache={},
    )

    _finalize_routing(agent, None, pool, "main-key")

    assert agent._credential_pool is pool


def test_pool_mutation_guard_rotates_the_models_own_pool(tmp_path, monkeypatch):
    """A 429 on the model's own pool rotates it; the URL-only guard called it a mismatch."""
    from agent.agent_runtime_helpers import recover_with_credential_pool
    from agent.error_classifier import FailoverReason

    _sibling_then_own(tmp_path, monkeypatch)
    agent = MagicMock()
    agent.provider, agent.base_url, agent.api_key = "custom", URL, "main-key"
    agent._credential_pool_entry_id = None
    pool = MagicMock()
    pool.provider = "custom:own"
    pool.entries.return_value = []
    pool.current.return_value = None
    next_entry = SimpleNamespace(id="own-2", runtime_api_key="own-2")
    pool.mark_exhausted_and_rotate.return_value = next_entry
    agent._credential_pool = pool

    recovered, _ = recover_with_credential_pool(
        agent, status_code=429, has_retried_429=True, classified_reason=FailoverReason.rate_limit,
    )

    assert recovered is True
    pool.mark_exhausted_and_rotate.assert_called_once()
    agent._swap_credential.assert_called_once_with(next_entry)


def _restorable_custom_agent(fallback_pool):
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider, agent.base_url = "m1", "openrouter", "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "fallback-key"
    agent._client_kwargs = {"api_key": "fallback-key", "base_url": "https://openrouter.ai/api/v1"}
    agent._credential_pool = fallback_pool
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.context_compressor = MagicMock()
    agent._primary_runtime = {
        "model": "m1", "provider": "custom", "base_url": URL, "api_mode": "chat_completions",
        "api_key": "main-key", "client_kwargs": {"api_key": "main-key", "base_url": URL},
        "use_prompt_caching": False, "use_native_cache_layout": False,
        "compressor_model": "m1", "compressor_base_url": URL, "compressor_api_key": "main-key",
        "compressor_provider": "custom", "compressor_context_length": 128000, "compressor_threshold_tokens": 0.8,
    }
    agent._create_openai_client = MagicMock(return_value=MagicMock())
    agent._apply_client_headers_for_base_url = MagicMock()
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    return agent


def test_restore_after_fallback_rebinds_the_models_own_pool_not_the_siblings(tmp_path, monkeypatch):
    """Probe Q8: after a cross-provider fallback the primary came back with the sibling's key."""
    from agent.credential_pool import CredentialPool

    _sibling_then_own(tmp_path, monkeypatch)
    agent = _restorable_custom_agent(CredentialPool("openrouter", []))

    assert agent._restore_primary_runtime() is True

    assert agent.provider == "custom"
    assert agent._credential_pool.provider == "custom:own"
    assert agent.api_key == "main-key"
    assert agent._client_kwargs["api_key"] == "main-key"


def test_restore_after_fallback_never_takes_a_siblings_key(tmp_path, monkeypatch):
    """Only a sibling on the URL: the primary keeps its own key and binds no pool."""
    from agent.credential_pool import CredentialPool

    _only_sibling(tmp_path, monkeypatch)
    agent = _restorable_custom_agent(CredentialPool("openrouter", []))

    assert agent._restore_primary_runtime() is True

    assert agent.api_key == "main-key"
    assert getattr(agent._credential_pool, "provider", None) != "custom:second"


# ── (B) bare custom + explicit base_url (direct alias) ─────────────────────────────────────


def test_direct_alias_with_explicit_key_skips_a_same_url_siblings_pool(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _only_sibling(tmp_path, monkeypatch, model_key=None)

    runtime = resolve_runtime_provider(requested="custom", explicit_api_key="alias-key", explicit_base_url=URL)

    assert runtime["api_key"] == "alias-key"
    assert runtime["source"] == "direct-alias"


def test_direct_alias_with_explicit_key_uses_its_own_pool(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _sibling_then_own(tmp_path, monkeypatch)

    runtime = resolve_runtime_provider(requested="custom", explicit_api_key="main-key", explicit_base_url=URL)

    assert runtime["api_key"] == "main-key"
    assert runtime["credential_pool"].provider == "custom:own"


def test_direct_alias_without_a_key_stays_url_only(tmp_path, monkeypatch):
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _only_sibling(tmp_path, monkeypatch, model_key=None)

    runtime = resolve_runtime_provider(requested="custom", explicit_base_url=URL)

    assert runtime["api_key"] == "second-key"
    assert runtime["credential_pool"].provider == "custom:second"


# ── (C) bare-custom delegated child ─────────────────────────────────────────────────────────


def _custom_parent(api_key="main-key", pool=None):
    return SimpleNamespace(provider="custom", base_url=URL, requested_provider="custom", api_key=api_key,
                           _credential_pool=pool)


def test_child_leases_the_pool_its_own_key_can_own(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _sibling_then_own(tmp_path, monkeypatch)

    pool = _resolve_child_credential_pool("custom", _custom_parent(), URL, effective_requested_provider="custom",
                                          effective_api_key="main-key")

    assert pool is not None and pool.provider == "custom:own"
    assert pool.select().access_token == "main-key"


def test_child_shares_the_parents_own_pool(tmp_path, monkeypatch):
    from agent.credential_pool import load_pool
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _sibling_then_own(tmp_path, monkeypatch)
    parent_pool = load_pool("custom:own")

    pool = _resolve_child_credential_pool("custom", _custom_parent(pool=parent_pool), URL,
                                          effective_requested_provider="custom", effective_api_key="main-key")

    assert pool is parent_pool


def test_child_never_leases_a_siblings_pool(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _only_sibling(tmp_path, monkeypatch)

    assert _resolve_child_credential_pool("custom", _custom_parent(), URL, effective_requested_provider="custom",
                                          effective_api_key="main-key") is None


def test_child_without_a_key_stays_url_only(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _only_sibling(tmp_path, monkeypatch, model_key=None)

    pool = _resolve_child_credential_pool("custom", _custom_parent(api_key=None), URL,
                                          effective_requested_provider="custom")

    assert pool is not None and pool.provider == "custom:second"


def test_built_child_inherits_the_parents_own_pool_not_the_siblings(tmp_path, monkeypatch):
    """End to end through ``_build_child_agent``: the child inherits the parent's endpoint and key."""
    from tools.delegate_tool import _build_child_agent

    _sibling_then_own(tmp_path, monkeypatch)
    parent = MagicMock()
    parent.provider, parent.base_url, parent.requested_provider = "custom", URL, "custom"
    parent.api_key, parent.api_mode, parent.model = "main-key", "chat_completions", "m1"
    parent._client_kwargs = {"api_key": "main-key", "base_url": URL}
    parent.client = None
    parent._credential_pool = None
    parent._delegate_depth = 0

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("run_agent.AIAgent", return_value=MagicMock()),
    ):
        child = _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None,
                                   max_iterations=3, parent_agent=parent, task_count=1)

    assert child._credential_pool.provider == "custom:own"
    assert child._credential_pool.select().access_token == "main-key"


# ── (D) end to end: a real AIAgent built from resolve_runtime_provider ──────────────────────
#
# A pool is the runtime's own when its key can be that entry's credential OR is one of the pool's
# rows. The row half is what keeps a single-entry setup working: a `hermes auth add` row (or the
# row a 429 rotated onto) differs from the entry's literal key, and there is no sibling at all.


def _manual_row(key, priority=0, row_id="man"):
    return {"id": row_id, "source": "manual", "auth_type": "api_key", "access_token": key,
            "base_url": URL, "label": row_id, "priority": priority}


def _one_entry_with_manual_row(tmp_path, monkeypatch, *, model_key, priority):
    """No sibling: one entry `Own` with literal key k1, plus a `hermes auth add` row k2."""
    _write_home(tmp_path, monkeypatch, {
        "model": _model(model_key),
        "custom_providers": [{"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"}],
    }, pools={"custom:own": [_manual_row("k2", priority)]})


def _built_agent(monkeypatch, requested="custom"):
    import run_agent
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from run_agent import AIAgent

    # run_agent froze HERMES_HOME at its first import, and file logging is process-global state an
    # earlier test may have pointed at another home; neither bears on which pool the agent binds.
    monkeypatch.setattr(run_agent, "_hermes_home", Path(os.environ["HERMES_HOME"]))
    monkeypatch.setattr("agent.agent_init._setup_logging", lambda agent: None)
    runtime = resolve_runtime_provider(requested=requested)
    agent = AIAgent(model="m1", provider=runtime["provider"], base_url=runtime["base_url"],
                    api_key=runtime["api_key"], api_mode=runtime.get("api_mode"),
                    credential_pool=runtime.get("credential_pool"),
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    return runtime, agent


def _fall_back_then_restore(agent):
    from agent.credential_pool import CredentialPool

    agent.provider, agent.base_url = "openrouter", "https://openrouter.ai/api/v1"
    agent.api_key = "fallback-key"
    agent._client_kwargs = {"api_key": "fallback-key", "base_url": "https://openrouter.ai/api/v1"}
    agent._credential_pool = CredentialPool("openrouter", [])
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    return agent._restore_primary_runtime()


def test_e1_sibling_first_agent_keeps_its_own_pool_at_init_and_after_restore(tmp_path, monkeypatch):
    """Probe E1 (Q8): the sibling is listed first. init_agent must hand the runtime's key to
    _finalize_routing, or the model's own pool is dropped; restore must not take the sibling's."""
    _sibling_then_own(tmp_path, monkeypatch)

    runtime, agent = _built_agent(monkeypatch)

    assert runtime["credential_pool"].provider == "custom:own"
    assert agent.api_key == "main-key"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"

    assert _fall_back_then_restore(agent) is True

    assert agent.api_key == "main-key"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"


def test_e2_keyless_model_on_a_manual_row_restores_its_own_pool(tmp_path, monkeypatch):
    """Probe E2: the keyless model resolves to the manual row k2; restore bound a pool named
    "custom" because a pool known only by name had no rows to find k2 in."""
    _one_entry_with_manual_row(tmp_path, monkeypatch, model_key=None, priority=0)
    _, agent = _built_agent(monkeypatch)
    assert agent.api_key == "k2"

    assert _fall_back_then_restore(agent) is True

    assert agent.api_key == "k2"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"


def test_e2_keyless_model_on_a_manual_row_shares_its_pool_with_a_child(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _one_entry_with_manual_row(tmp_path, monkeypatch, model_key=None, priority=0)
    _, agent = _built_agent(monkeypatch)

    child_pool = _resolve_child_credential_pool("custom", agent, agent.base_url, effective_requested_provider=None,
                                                effective_api_key=agent.api_key)

    assert child_pool is agent._credential_pool
    assert child_pool.provider == "custom:own"


def test_e3_rotated_parent_shares_its_pool_with_a_child(tmp_path, monkeypatch):
    """Probe E3: model key k1 is the entry's; a 429 rotated the parent onto the manual row k2."""
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _one_entry_with_manual_row(tmp_path, monkeypatch, model_key="k1", priority=5)
    _, agent = _built_agent(monkeypatch)
    pool = agent._credential_pool
    assert pool.provider == "custom:own"
    agent.api_key = "k2"  # mark_exhausted_and_rotate + _swap_credential

    child_pool = _resolve_child_credential_pool("custom", agent, agent.base_url, effective_requested_provider=None,
                                                effective_api_key="k2")

    assert child_pool is pool


def test_e4_built_child_of_a_rotated_parent_shares_its_pool(tmp_path, monkeypatch):
    """Probe E4: E3 end to end through ``_build_child_agent``."""
    from tools.delegate_tool import _build_child_agent

    _one_entry_with_manual_row(tmp_path, monkeypatch, model_key="k1", priority=5)
    _, agent = _built_agent(monkeypatch)
    pool = agent._credential_pool
    agent.api_key = "k2"
    agent._client_kwargs["api_key"] = "k2"

    with (
        patch("tools.delegate_tool._load_config", return_value={}),
        patch("run_agent.AIAgent", return_value=MagicMock()),
    ):
        child = _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None,
                                   max_iterations=3, parent_agent=agent, task_count=1)

    assert child._credential_pool is pool


def test_e5_init_keeps_the_pool_a_manual_row_key_came_from(tmp_path, monkeypatch):
    """Probe E5: _finalize_routing keeps the pool whose row the runtime key is."""
    _one_entry_with_manual_row(tmp_path, monkeypatch, model_key=None, priority=0)

    runtime, agent = _built_agent(monkeypatch)

    assert runtime["credential_pool"].provider == "custom:own"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"


def test_child_on_another_url_never_shares_a_parent_pool_that_holds_the_same_key(tmp_path, monkeypatch):
    """One key reused on two endpoints: the parent's pool (entry on URL2) holds the child's key as a
    row, but it is no entry on the child's URL, so row membership must not make it the child's."""
    from agent.credential_pool import credential_pool_matches_provider, load_pool
    from tools.delegate_tool_config import _resolve_child_credential_pool

    url2 = "https://other.example.test/v1"
    _write_home(tmp_path, monkeypatch, {
        "model": _model(None),
        "custom_providers": [
            {"name": "Own", "base_url": URL, "model": "m1"},
            {"name": "Other", "base_url": url2, "api_key": "k1", "model": "m1"},
        ],
    }, pools={"custom:other": [dict(_manual_row("shared-key"), base_url=url2)]})
    parent_pool = load_pool("custom:other")
    parent = SimpleNamespace(provider="custom", base_url=url2, requested_provider="custom", api_key="shared-key",
                             _credential_pool=parent_pool)

    assert not credential_pool_matches_provider(parent_pool, "custom", base_url=URL, owner_api_key="shared-key")
    child_pool = _resolve_child_credential_pool("custom", parent, URL, effective_requested_provider="custom",
                                                effective_api_key="shared-key")

    assert child_pool is not parent_pool
    assert getattr(child_pool, "provider", None) != "custom:other"


# ── (E) ownership order: declared credential, then a pool row, then a credential-less entry ──
#
# A credential-less entry listed first used to claim ANY key, so the model ran on that entry's
# rows. And a key a sibling declares can also be a row of the primary's own pool: the key alone
# cannot settle that, so restore rebinds to the pool the primary ran from.


def _pool_sources(pool_key):
    from agent.credential_pool import load_pool

    return {(entry.source, entry.access_token) for entry in load_pool(pool_key).entries()}


def _keyless_sibling_first(tmp_path, monkeypatch):
    """X5: keyless Sib (a `hermes auth add` row ks) listed first, Own with literal k1 second,
    model key k1."""
    _write_home(tmp_path, monkeypatch, {
        "model": _model("k1"),
        "custom_providers": [
            {"name": "Sib", "base_url": URL, "model": "m1"},
            {"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"},
        ],
    }, pools={"custom:sib": [_manual_row("ks", row_id="sib-ks")]})


def _two_keyless_entries(tmp_path, monkeypatch, *, stale_model_config=False):
    """X7: keyless A (row a1) and keyless B (row b1), model key b1."""
    a_rows = [_manual_row("a1", row_id="a-a1")]
    if stale_model_config:
        # An earlier version seeded the model's key into the FIRST credential-less entry.
        a_rows.append({"id": "stale", "source": "model_config", "auth_type": "api_key", "access_token": "b1",
                       "base_url": URL, "label": "model_config", "priority": 1})
    _write_home(tmp_path, monkeypatch, {
        "model": _model("b1"),
        "custom_providers": [
            {"name": "A", "base_url": URL, "model": "m1"},
            {"name": "B", "base_url": URL, "model": "m1"},
        ],
    }, pools={"custom:a": a_rows, "custom:b": [_manual_row("b1", row_id="b-b1")]})


def test_x5_keyless_sibling_first_never_claims_the_key_an_entry_declares(tmp_path, monkeypatch):
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _keyless_sibling_first(tmp_path, monkeypatch)

    runtime, agent = _built_agent(monkeypatch)

    assert runtime["credential_pool"].provider == "custom:own"
    assert agent.api_key == "k1"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"
    child_pool = _resolve_child_credential_pool("custom", agent, agent.base_url, effective_requested_provider=None,
                                                effective_api_key=agent.api_key)
    assert getattr(child_pool, "provider", None) == "custom:own"

    assert _fall_back_then_restore(agent) is True

    assert agent.api_key == "k1"
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"


def test_x5_model_config_is_seeded_into_the_declaring_entry_not_the_keyless_one(tmp_path, monkeypatch):
    _keyless_sibling_first(tmp_path, monkeypatch)

    assert ("model_config", "k1") in _pool_sources("custom:own")
    assert all(source != "model_config" for source, _ in _pool_sources("custom:sib"))


def test_x7_the_entry_holding_the_key_as_a_row_beats_an_earlier_keyless_entry(tmp_path, monkeypatch):
    from agent.credential_pool import resolve_runtime_pool_key
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _two_keyless_entries(tmp_path, monkeypatch)

    runtime, agent = _built_agent(monkeypatch)

    assert runtime["credential_pool"].provider == "custom:b"
    assert agent.api_key == "b1"
    assert resolve_runtime_pool_key("custom", URL, owner_api_key="b1") == "custom:b"
    child_pool = _resolve_child_credential_pool("custom", agent, agent.base_url, effective_requested_provider=None,
                                                effective_api_key=agent.api_key)
    assert getattr(child_pool, "provider", None) == "custom:b"

    assert _fall_back_then_restore(agent) is True

    assert agent.api_key == "b1"
    assert getattr(agent._credential_pool, "provider", None) == "custom:b"


def test_x7_a_stale_model_config_row_is_no_claim_on_the_key(tmp_path, monkeypatch):
    """The model_config row an earlier version left in A must not make A the owner again; it is
    pruned once B owns the key."""
    from agent.credential_pool import custom_pool_keys_for_owner_key

    _two_keyless_entries(tmp_path, monkeypatch, stale_model_config=True)

    assert custom_pool_keys_for_owner_key(URL, "b1") == ["custom:b"]
    assert ("model_config", "b1") not in _pool_sources("custom:a")
    assert ("model_config", "b1") in _pool_sources("custom:b")


def test_credential_less_entry_still_owns_a_key_nothing_better_owns(tmp_path, monkeypatch):
    """#100413 kept: with no declaring entry and no row holding it, the keyless entry owns the key,
    even behind a sibling that declares another."""
    from agent.credential_pool import custom_pool_keys_for_owner_key

    _sibling_then_own(tmp_path, monkeypatch)

    assert custom_pool_keys_for_owner_key(URL, "main-key") == ["custom:own"]
    assert ("model_config", "main-key") in _pool_sources("custom:own")


def _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, *, model_key, own_priority):
    """X2b: Own (literal k1) first, Sib (literal ks) second; Own's pool holds ks as a `hermes auth
    add` row and Sib's pool has a further row ks2."""
    _write_home(tmp_path, monkeypatch, {
        "model": _model(model_key),
        "custom_providers": [
            {"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"},
            {"name": "Sib", "base_url": URL, "api_key": "ks", "model": "m1"},
        ],
    }, pools={"custom:own": [_manual_row("ks", own_priority, row_id="own-ks")],
              "custom:sib": [_manual_row("ks2", 0, row_id="sib-ks2")]})


@pytest.mark.parametrize(("model_key", "own_priority"), [("k1", 5), ("k1", 0), (None, 0)])
def test_x2b_restore_rebinds_the_pool_the_primary_ran_from(tmp_path, monkeypatch, model_key, own_priority):
    """The primary runs on ks from its own pool. ks is also the sibling's declared key, so the
    owner-key lookup names the sibling's pool, whose rebind swapped in ks2."""
    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key=model_key, own_priority=own_priority)
    _, agent = _built_agent(monkeypatch)
    assert getattr(agent._credential_pool, "provider", None) == "custom:own"
    init_key = agent.api_key

    assert _fall_back_then_restore(agent) is True

    assert (getattr(agent._credential_pool, "provider", None), agent.api_key) == ("custom:own", init_key)
    assert agent._client_kwargs["api_key"] == init_key
    assert agent._primary_runtime["credential_pool_key"] == "custom:own"


def test_restore_ignores_a_recorded_pool_that_no_longer_owns_the_key(tmp_path, monkeypatch):
    """A recorded pool is used only while it holds or can own the primary's key; else the owner-key
    lookup decides."""
    from agent.credential_pool import CredentialPool

    _sibling_then_own(tmp_path, monkeypatch)
    agent = _restorable_custom_agent(CredentialPool("openrouter", []))
    agent._primary_runtime["credential_pool_key"] = "custom:second"

    assert agent._restore_primary_runtime() is True

    assert agent._credential_pool.provider == "custom:own"
    assert agent.api_key == "main-key"


def test_switch_model_snapshot_records_the_pool(tmp_path, monkeypatch):
    from agent.agent_runtime_helpers import _build_primary_runtime_snapshot
    from agent.credential_pool import CredentialPool

    agent = _restorable_custom_agent(CredentialPool("custom:own", []))
    agent.request_overrides, agent.reasoning_config, agent.requested_provider = {}, None, "custom"

    assert _build_primary_runtime_snapshot(agent, "chat_completions")["credential_pool_key"] == "custom:own"
    agent._credential_pool = None
    assert _build_primary_runtime_snapshot(agent, "chat_completions")["credential_pool_key"] is None


# ── (F) delegation and restore-lookup guards ────────────────────────────────────────────────

UNREG1 = "https://unreg-one.example.test/v1"
UNREG2 = "https://unreg-two.example.test/v1"


def test_restore_lookup_finds_the_own_pool_by_its_row_behind_a_sibling(tmp_path, monkeypatch):
    """K2: sibling first; Own's literal is k1; the primary is on Own's `hermes auth add` row k2."""
    from agent.credential_pool import resolve_runtime_pool_key

    _write_home(tmp_path, monkeypatch, {
        "model": _model(None),
        "custom_providers": [
            {"name": "Sib", "base_url": URL, "api_key": "ks", "model": "m1"},
            {"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"},
        ],
    }, pools={"custom:own": [_manual_row("k2", row_id="own-k2")]})

    assert resolve_runtime_pool_key("custom", URL, owner_api_key="k2") == "custom:own"


def test_child_on_a_row_of_the_parents_pool_shares_it_though_a_sibling_declares_that_key(tmp_path, monkeypatch):
    """K4: the parent runs on its entry key k1; the child is on ks, a row of the parent's pool that
    is ALSO a sibling's literal key. The share branch keeps the child on the parent's pool."""
    from agent.credential_pool import load_pool
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _write_home(tmp_path, monkeypatch, {
        "model": _model("k1"),
        "custom_providers": [
            {"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"},
            {"name": "Sib", "base_url": URL, "api_key": "ks", "model": "m1"},
        ],
    }, pools={"custom:own": [_manual_row("ks", 5, row_id="own-ks")]})
    parent_pool = load_pool("custom:own")

    child_pool = _resolve_child_credential_pool("custom", _custom_parent(api_key="k1", pool=parent_pool), URL,
                                                effective_requested_provider="custom", effective_api_key="ks")

    assert child_pool is parent_pool


def test_a_parent_pool_named_plain_custom_is_never_shared_across_endpoints(tmp_path, monkeypatch):
    """K5 / #7833: a parent on an unregistered endpoint holds a pool named plain "custom"; a child
    on another unregistered endpoint keeps its fixed credential."""
    from agent.credential_pool import CredentialPool, PooledCredential
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _write_home(tmp_path, monkeypatch, {"model": {"default": "m1", "provider": "custom", "base_url": UNREG1}})
    parent_pool = CredentialPool("custom", [PooledCredential.from_dict("custom", dict(_manual_row("p1"),
                                                                                       base_url=UNREG1))])
    parent = SimpleNamespace(provider="custom", base_url=UNREG1, requested_provider="custom", api_key="p1",
                             _credential_pool=parent_pool)

    child_pool = _resolve_child_credential_pool("custom", parent, UNREG2, effective_requested_provider="custom",
                                                effective_api_key="c1")

    assert child_pool is not parent_pool


def test_named_child_leases_its_named_pool_not_the_one_its_key_would_name(tmp_path, monkeypatch):
    """K6 / #45763: a named child 'second' matches by name. By key alone ("fresh-key": declared by
    nobody, held by no row) the credential-less Own listed first would own it."""
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _write_home(tmp_path, monkeypatch, {
        "model": {"default": "m1", "provider": "openrouter"},
        "custom_providers": [
            {"name": "Own", "base_url": URL, "model": "m1"},
            {"name": "Second", "base_url": URL, "api_key": "second-key", "model": "m1"},
        ],
    })
    parent = SimpleNamespace(provider="openrouter", base_url="https://openrouter.ai/api/v1",
                             requested_provider="openrouter", api_key="or", _credential_pool=None)

    child_pool = _resolve_child_credential_pool("custom", parent, URL, effective_requested_provider="second",
                                                effective_api_key="fresh-key")

    assert getattr(child_pool, "provider", None) == "custom:second"


def test_child_with_a_siblings_override_key_never_leases_the_parents_pool(tmp_path, monkeypatch):
    """K12b: delegation.base_url is the parent's URL and delegation.api_key the sibling's key. The
    parent's key names the parent's pool, so it is not the child's (its rows would replace the
    override key)."""
    from agent.credential_pool import load_pool
    from tools.delegate_tool_config import _resolve_child_credential_pool

    _sibling_then_own(tmp_path, monkeypatch)
    parent_pool = load_pool("custom:own")

    child_pool = _resolve_child_credential_pool("custom", _custom_parent(pool=parent_pool), URL,
                                                effective_requested_provider="custom", effective_api_key="second-key")

    assert child_pool is not parent_pool


# ── (G) the pinned pool: preferred over any attached pool, never plain "custom" ─────────────
#
# The recorded pool used to be consulted only when the attached pool failed the owner match. A
# fallback to the same-URL sibling leaves ITS pool attached, and that pool also matches (the
# sibling declares the snapshot key), so its row ks2 was swapped in.


def _drop_row(pool_key, key):
    auth = Path(os.environ["HERMES_HOME"]) / "auth.json"
    store = json.loads(auth.read_text())
    store["credential_pool"][pool_key] = [row for row in store["credential_pool"].get(pool_key, [])
                                          if row.get("access_token") != key]
    auth.write_text(json.dumps(store))


@pytest.mark.parametrize("fallback_provider", ["custom:sib", "sib"])
def test_restore_after_a_fallback_to_the_same_url_sibling_goes_back_to_the_pinned_pool(
        tmp_path, monkeypatch, fallback_provider):
    """S1r: a real fallback to the sibling (``custom:sib`` attaches its pool, ``sib`` none)."""
    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    _, agent = _built_agent(monkeypatch)
    init_key = agent.api_key
    agent._fallback_chain = [{"provider": fallback_provider, "model": "m2"}]
    agent._fallback_index = 0
    assert agent._try_activate_fallback() is True
    agent._rate_limited_until = 0
    agent._replace_primary_openai_client = MagicMock(return_value=True)

    assert agent._restore_primary_runtime() is True

    assert (getattr(agent._credential_pool, "provider", None), agent.api_key) == ("custom:own", init_key)
    assert agent._client_kwargs["api_key"] == init_key


def test_reset_gate_reads_the_pinned_pool_not_the_attached_siblings(tmp_path, monkeypatch):
    """Every row of the primary's own pool is benched past now: restore waits, though the attached
    sibling's pool (after a fallback to it) has a free row."""
    import time

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    _, agent = _built_agent(monkeypatch)
    auth = Path(os.environ["HERMES_HOME"]) / "auth.json"
    store = json.loads(auth.read_text())
    assert {row["source"] for row in store["credential_pool"]["custom:own"]} >= {"manual", "config:Own"}
    for row in store["credential_pool"]["custom:own"]:
        row.update(last_status="exhausted", last_status_at=time.time(), last_error_code=429,
                   last_error_reset_at=time.time() + 3600)
    auth.write_text(json.dumps(store))
    agent._fallback_chain = [{"provider": "custom:sib", "model": "m2"}]
    agent._fallback_index = 0
    assert agent._try_activate_fallback() is True
    agent._rate_limited_until = 0

    assert agent._restore_primary_runtime() is False

    assert (agent.provider, agent._credential_pool.provider) == ("custom:sib", "custom:sib")


def test_restore_after_a_one_turn_switch_to_the_same_url_sibling_goes_back_to_the_pinned_pool(tmp_path, monkeypatch):
    """S1o: ``/model --provider custom:sib --once`` attaches the sibling's pool for one turn."""
    import copy

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    _, agent = _built_agent(monkeypatch)
    init_key, saved = agent.api_key, copy.deepcopy(agent._primary_runtime)
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    agent.switch_model("m2", "custom:sib", api_key="ks", base_url=URL, api_mode="chat_completions")
    assert agent._credential_pool.provider == "custom:sib"
    agent._primary_runtime, agent._fallback_activated, agent._rate_limited_until = saved, True, 0

    assert agent._restore_primary_runtime() is True

    assert (agent._credential_pool.provider, agent.api_key, agent.base_url) == ("custom:own", init_key, URL)


def test_restore_keeps_the_pin_after_its_row_is_removed_while_it_owns_the_configured_key(tmp_path, monkeypatch):
    """S2: ``hermes auth remove`` of the row ks the primary ran on. ks is the sibling's declared key,
    but the pinned pool is still that of the entry declaring the model's configured key k1."""
    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    _, agent = _built_agent(monkeypatch)
    assert agent.api_key == "ks"
    _drop_row("custom:own", "ks")

    assert _fall_back_then_restore(agent) is True

    assert (agent._credential_pool.provider, agent.api_key) == ("custom:own", "k1")


def test_restore_never_follows_a_plain_custom_pin_to_another_endpoint(tmp_path, monkeypatch):
    """S9b: switch_model onto bare custom attaches pool "custom", which holds a row of ANOTHER
    endpoint; the snapshot records it. Restore must not adopt that row's key and base_url."""
    _write_home(tmp_path, monkeypatch, {
        "model": {"default": "m0", "provider": "openrouter"},
        "custom_providers": [{"name": "Own", "base_url": URL, "api_key": "k1", "model": "m1"}],
    }, pools={"custom": [dict(_manual_row("p1"), base_url=UNREG1)]})
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    _, agent = _built_agent(monkeypatch, requested="openrouter")
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    agent.switch_model("m1", "custom", api_key="k1", base_url=URL, api_mode="chat_completions")

    assert _fall_back_then_restore(agent) is True

    assert (agent.base_url, agent.api_key) == (URL, "k1")
    assert agent._credential_pool.provider == "custom:own"


def test_restore_never_adopts_an_attached_pool_row_of_another_endpoint(tmp_path, monkeypatch):
    """No pin: the attached pool "custom" matches bare custom by name, but its best row serves
    another endpoint; the primary keeps its own key and URL."""
    from agent.credential_pool import load_pool

    _sibling_then_own(tmp_path, monkeypatch)
    home = Path(os.environ["HERMES_HOME"])
    store = json.loads((home / "auth.json").read_text())
    store["credential_pool"]["custom"] = [dict(_manual_row("p1"), base_url=UNREG1)]
    (home / "auth.json").write_text(json.dumps(store))
    agent = _restorable_custom_agent(load_pool("custom"))

    assert agent._restore_primary_runtime() is True

    assert (agent.base_url, agent.api_key) == (URL, "main-key")


def test_restore_rotates_off_an_exhausted_pinned_row(tmp_path, monkeypatch):
    """PA7: the row ks the primary ran on is exhausted while on fallback; restore selects k1 from the
    same pinned pool, never a sibling's row."""
    import time

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    _, agent = _built_agent(monkeypatch)
    assert agent.api_key == "ks"
    auth = Path(os.environ["HERMES_HOME"]) / "auth.json"
    store = json.loads(auth.read_text())
    for row in store["credential_pool"]["custom:own"]:
        if row.get("access_token") == "ks":
            row.update(last_status="exhausted", last_status_at=time.time(), last_error_code=429,
                       last_error_reset_at=time.time() + 3600)
    auth.write_text(json.dumps(store))

    assert _fall_back_then_restore(agent) is True

    assert (agent._credential_pool.provider, agent.api_key) == ("custom:own", "k1")


def test_pin_is_valid_only_for_a_configured_entry_on_the_url_that_still_owns_a_key(tmp_path, monkeypatch):
    from agent.credential_pool import custom_pool_pin_serves_primary

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)

    assert custom_pool_pin_serves_primary("custom:own", "custom", URL, "ks")
    assert custom_pool_pin_serves_primary("custom:sib", "custom", URL, "ks2")
    assert not custom_pool_pin_serves_primary("custom", "custom", URL, "k1")  # plain: may hold other endpoints
    assert not custom_pool_pin_serves_primary("custom:own", "custom", UNREG1, "ks")  # entry on another URL
    assert not custom_pool_pin_serves_primary("custom:own", "custom:own", URL, "ks")  # named runtime: by name
    _drop_row("custom:own", "ks")
    # ks is now only the sibling's; the model's configured key k1 still makes Own the owner.
    assert custom_pool_pin_serves_primary("custom:own", "custom", URL, "ks")


def test_a_sibling_pin_is_refused_when_neither_key_is_its_own(tmp_path, monkeypatch):
    from agent.credential_pool import custom_pool_pin_serves_primary

    _sibling_then_own(tmp_path, monkeypatch)

    assert not custom_pool_pin_serves_primary("custom:second", "custom", URL, "main-key")
    assert custom_pool_pin_serves_primary("custom:own", "custom", URL, "main-key")


def test_a_configured_key_counts_only_from_a_model_block_on_the_primary_url(tmp_path, monkeypatch):
    """The model: block is on another endpoint; its key (the sibling's literal) says nothing about
    which pool on the primary URL is the primary's."""
    from agent.credential_pool import custom_pool_pin_serves_primary

    _write_home(tmp_path, monkeypatch, {
        "model": {"default": "m1", "provider": "custom", "base_url": UNREG1, "api_key": "second-key"},
        "custom_providers": [
            {"name": "Second", "base_url": URL, "api_key": "second-key", "model": "m1"},
            {"name": "Own", "base_url": URL, "model": "m1"},
        ],
    })

    assert not custom_pool_pin_serves_primary("custom:second", "custom", URL, "main-key")


# ── (H) ownership order details ─────────────────────────────────────────────────────────────


def test_a_declared_key_beats_a_row_of_an_earlier_entry(tmp_path, monkeypatch):
    """PA17 (tier 1 before tier 2): the model's key ks is Sib's literal and also a row of Own's pool,
    Own listed first. Sib owns it; the model runs from Sib's pool, and within that pool the row is
    picked by the pool's own priority (ks2 here), as for every pool."""
    from agent.credential_pool import custom_pool_keys_for_owner_key, resolve_runtime_pool_key

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="ks", own_priority=5)

    assert custom_pool_keys_for_owner_key(URL, "ks") == ["custom:sib"]
    assert resolve_runtime_pool_key("custom", URL, owner_api_key="ks") == "custom:sib"
    _, agent = _built_agent(monkeypatch)
    assert (agent._credential_pool.provider, agent.api_key) == ("custom:sib", "ks2")


def test_a_row_held_in_the_legacy_pool_puts_that_pool_first(tmp_path, monkeypatch):
    """Tier 2 over one entry's two identities (``providers.<slug>`` and legacy ``custom:<name>``):
    the identity whose pool holds the key comes first, so the runtime binds that pool."""
    from agent.credential_pool import custom_pool_keys_for_owner_key, resolve_runtime_pool_key

    _write_home(tmp_path, monkeypatch, {
        "model": _model("k2"),
        "providers": {"ownslug": {"name": "Own Display", "base_url": URL, "api_key": "k1", "model": "m1"}},
    }, pools={"custom:own-display": [_manual_row("k2", row_id="legacy-k2")]})

    assert custom_pool_keys_for_owner_key(URL, "k2") == ["custom:own-display", "ownslug"]
    assert custom_pool_keys_for_owner_key(URL, "k1") == ["ownslug", "custom:own-display"]
    assert resolve_runtime_pool_key("custom", URL, owner_api_key="k2") == "custom:own-display"


def test_pool_match_reads_the_rows_of_a_pooled_credentials_pool(tmp_path, monkeypatch):
    """A7: a pool row (not a pool) is matched by its pool's persisted rows: ks is a row of Own's pool
    though Sib declares it."""
    from agent.credential_pool import credential_pool_matches_provider, load_pool

    _own_first_row_is_the_siblings_key(tmp_path, monkeypatch, model_key="k1", own_priority=5)
    k1_row = next(e for e in load_pool("custom:own").entries() if e.access_token == "k1")

    assert credential_pool_matches_provider(k1_row, "custom", base_url=URL, owner_api_key="ks")
    _drop_row("custom:own", "ks")
    assert not credential_pool_matches_provider(k1_row, "custom", base_url=URL, owner_api_key="ks")
