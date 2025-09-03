#!/usr/bin/env python3
"""
Diagnostic helper to inspect UnifiedCacheManager stats and recent cache entries.
Run inside project root (or inside Docker) to get quick visibility into cache state.
"""

from __future__ import annotations

import json

from src.core.cache_manager import UnifiedCacheManager


def main():
    cm = UnifiedCacheManager()
    stats = cm.get_cache_stats()
    print("Cache stats:")
    print(json.dumps(stats, indent=2))

    # List recent data cache entries
    print("\nRecent data cache entries (up to 20):")
    entries = cm._find_entries("data")  # internal helper
    entries_sorted = sorted(entries, key=lambda e: e.last_accessed, reverse=True)
    for e in entries_sorted[:20]:
        print(
            f"- key={e.key} source={e.source} symbol={e.symbol} interval={e.interval} created_at={e.created_at.isoformat()} size_bytes={e.size_bytes}"
        )


if __name__ == "__main__":
    main()
