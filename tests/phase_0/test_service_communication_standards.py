"""Phase 0 Task 5 - Service-to-service communication standards.

The Phase 0 roadmap requires explicit guidance for:
- gRPC request/response services
- Kafka or ZeroMQ event streaming
- synchronous vs asynchronous control-plane traffic

These tests enforce a minimum content bar on the standards document so that
Phase 1 transport work in `shared_lib.transport` can reference a stable
architectural contract rather than informal tribal knowledge.
"""

from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("docs") / "architecture" / "service-communication-standards.md"


REQUIRED_SECTIONS = (
    "## Scope",
    "## Transport Matrix",
    "## Synchronous Control-Plane Traffic",
    "## Asynchronous and Streaming Traffic",
    "## Schema and Contract Rules",
    "## Failure Modes",
    "## Security and Authentication",
    "## Enforcement",
)


REQUIRED_TRANSPORTS = ("gRPC", "Kafka", "ZeroMQ", "HTTP", "REST")


REQUIRED_CONCEPTS = (
    # Every standard must address these concepts.
    "protobuf",
    "idempoten",  # matches idempotent / idempotency
    "retry",
    "backpressure",
    "trace_id",
    "otel",  # OpenTelemetry
    "authenticat",
    "mtls",
)


def _read(repo_root: Path) -> str:
    path = repo_root / DOC_PATH
    assert path.is_file(), f"Service communication doc missing at {DOC_PATH}"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge case 1: every required section must exist.
# ---------------------------------------------------------------------------


def test_doc_has_all_required_sections(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [h for h in REQUIRED_SECTIONS if h not in text]
    assert not missing, f"Service-comm doc missing sections: {missing}"


# ---------------------------------------------------------------------------
# Edge case 2: every required transport option must be named.
# ---------------------------------------------------------------------------


def test_doc_names_all_required_transports(repo_root: Path) -> None:
    text = _read(repo_root)
    missing = [t for t in REQUIRED_TRANSPORTS if t not in text]
    assert not missing, f"Service-comm doc missing transports: {missing}"


# ---------------------------------------------------------------------------
# Edge case 3: every required concept (idempotency, retry, backpressure,
# tracing, authentication) must be addressed.
# ---------------------------------------------------------------------------


def test_doc_addresses_required_concepts(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    missing = [c for c in REQUIRED_CONCEPTS if c.lower() not in text]
    assert not missing, f"Service-comm doc missing concepts: {missing}"


# ---------------------------------------------------------------------------
# Edge case 4: the sync-vs-async split must be explicit enough to be
# actionable. We require both words `synchronous` and `asynchronous` and at
# least one explicit routing rule (e.g. "use ... when ...").
# ---------------------------------------------------------------------------


def test_doc_has_actionable_sync_async_rules(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "synchronous" in text and "asynchronous" in text, (
        "Service-comm doc must cover both synchronous and asynchronous paths"
    )
    # Require at least one routing-rule phrase so the guidance is not abstract.
    assert (
        " use " in text or " prefer " in text or " default " in text
    ), "Service-comm doc must include at least one explicit routing rule"


# ---------------------------------------------------------------------------
# Edge case 5: the HFT boundary must be referenced so the transport rules
# map onto ADR-0003.
# ---------------------------------------------------------------------------


def test_doc_references_hft_boundary(repo_root: Path) -> None:
    text = _read(repo_root).lower()
    assert "hft" in text, (
        "Service-comm doc must reference the HFT boundary (ADR-0003)"
    )


# ---------------------------------------------------------------------------
# Edge case 6: the transport matrix must contain an actual markdown table.
# ---------------------------------------------------------------------------


def test_doc_contains_transport_matrix_table(repo_root: Path) -> None:
    text = _read(repo_root)
    # Locate the section body and check for a markdown table header separator.
    idx = text.find("## Transport Matrix")
    assert idx >= 0, "Transport Matrix section must exist"
    section = text[idx : idx + 2000]
    assert "|" in section and "---" in section, (
        "Transport Matrix must include a markdown table with a separator row"
    )
