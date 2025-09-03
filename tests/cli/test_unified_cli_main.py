from __future__ import annotations

import src.cli.unified_cli as cli


def test_main_no_args_returns_1():
    assert cli.main([]) == 1


def test_main_unknown_returns_2():
    assert cli.main(["not-a-subcommand"]) == 2


def test_main_routes_to_collection(monkeypatch):
    called = {"v": False}

    def fake_handle(argv):
        called["v"] = True
        return 0

    monkeypatch.setattr(cli, "handle_collection_run", fake_handle)
    rc = cli.main(["collection", "bonds_core", "--dry-run"])  # args forwarded
    assert rc == 0
    assert called["v"] is True
