"""Import-light policy helpers, not dispatch or complete discovery coverage.

The full native spawn schema is dropped on ACP; owned controls are separately
injected. Real dispatch/catalog regressions live in test_native_delegate_policy.
"""

from __future__ import annotations

import pytest

from toolsets import acp_delegate_refusal, session_platform_tool_drops


def test_acp_drops_delegate_task_other_platforms_keep_it():
    acp = session_platform_tool_drops("acp")
    assert "delegate_task" in acp
    for platform in ("telegram", "cli", "desktop", None):
        assert "delegate_task" not in session_platform_tool_drops(platform)


def test_acp_still_drops_other_platform_gated_toolsets():
    # Existing platform gating must survive the new drop (catalog is desktop-only).
    assert "manage_catalog" in session_platform_tool_drops("acp")
    assert "manage_catalog" not in session_platform_tool_drops("desktop")


def test_dispatch_refusal_helper(monkeypatch):
    # acp source (platform OR session env) → policy refusal; others → None.
    assert acp_delegate_refusal("acp") is not None
    assert "disabled by policy" in acp_delegate_refusal("acp")
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "acp")
    assert (
        acp_delegate_refusal("telegram") is not None
    )  # session env wins over platform
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    assert acp_delegate_refusal("telegram") is None
    assert acp_delegate_refusal("cli") is None
    assert acp_delegate_refusal(None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
