from __future__ import annotations

import pandas as pd

from src.core.cache_manager import UnifiedCacheManager


def test_redis_helpers_no_redis_installed(monkeypatch):
    cm = UnifiedCacheManager()
    # Ensure client is None
    cm.redis_client = None
    assert cm.get_recent_overlay_from_redis("TLT", "1d") is None
    # set should not raise
    cm.set_recent_overlay_to_redis("TLT", "1d", pd.DataFrame())
