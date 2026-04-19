# Service-to-Service Communication Standards

- Status: Accepted (Phase 0)
- Source ADRs: ADR-0001, ADR-0003
- Applies to: every cross-package or cross-service call inside the
  quant-system monorepo, including Python ↔ Python, Python ↔ native, and
  Python ↔ web-control-plane traffic.
- Enforcement: ruff lint rules, CI contract tests, and the Phase 5
  code_reviewer agent.

## Scope

These standards cover how services and domain packages talk to each other.
They do not cover:

- in-process function calls inside the same package (use normal Python).
- connectivity to external brokers, exchanges, or RPC providers (see
  `trading_system/shared_gateways/` and ADR-0005).

Everything else — research ↔ data platform, backtest engine ↔ OMS, agents ↔
control plane, control plane ↔ OMS, mid-freq engine ↔ HFT engine — must
honour this document.

## Transport Matrix

| Use case                                         | Transport              | Format            | Delivery | Notes                                                                 |
|--------------------------------------------------|------------------------|-------------------|----------|-----------------------------------------------------------------------|
| Synchronous request/response between services    | gRPC over HTTP/2       | protobuf          | Unary    | Default for Python ↔ Python and Python ↔ native RPC.                  |
| Browser ↔ web_control_plane backend              | HTTPS REST + JSON      | pydantic-derived  | Unary    | Browser never talks gRPC directly; REST stays inside the backend API. |
| Low-latency streaming (market data, signals)     | ZeroMQ PUB/SUB or PUSH | binary / protobuf | At-least once / fire-and-forget | Used on the HFT side of ADR-0003 and for intra-node fanout.          |
| Durable event streaming (fills, audit, OMS logs) | Kafka                  | protobuf (Avro OK)| At-least once, ordered per key | Required when the consumer needs replay or cross-team consumption.    |
| Long-running job control (backfills, DAGs)       | Airflow + REST + Kafka | JSON + protobuf   | Async    | DAG triggers are REST; run results are Kafka events.                  |
| Fire-and-forget notifications                    | HTTPS webhooks / Slack | JSON              | Async    | Not used as a source of truth.                                        |
| Python ↔ native HFT handoff                      | shared memory + ZeroMQ | binary            | Bounded  | Python is strictly on the send side; see ADR-0003.                    |

The matrix is normative: if a new service needs a transport that is not
listed, the proposal must update this document and any relevant ADR before
implementation.

## Synchronous Control-Plane Traffic

Use synchronous gRPC when **all** of the following hold:

- the caller cannot make progress without the response;
- the call is expected to complete in under 500 ms at the 99th percentile;
- the call is idempotent or guarded by an idempotency key in the request
  envelope.

Default rules for synchronous calls:

- **Idempotency**: every mutating RPC must accept an `idempotency_key`
  field. Duplicate keys must return the original response. This is the
  contract, not a library feature.
- **Retries**: clients retry with exponential backoff only on retriable
  status codes (`UNAVAILABLE`, `DEADLINE_EXCEEDED`). Max 3 attempts,
  jittered.
- **Deadlines**: every RPC carries a server-side deadline. No unbounded
  calls.
- **Tracing**: every request propagates `trace_id` and `span_id` via gRPC
  metadata or REST headers (`traceparent`). OTel context propagation is
  mandatory.
- **Auth**: internal RPC traffic prefers mTLS between services; REST traffic
  from the browser uses authenticated sessions enforced by the
  web_control_plane backend. No service exposes unauthenticated mutating
  endpoints.

## Asynchronous and Streaming Traffic

Use Kafka when the message is a **fact** that may have multiple consumers or
must be replayable (fills, audit events, approvals, anomaly events, agent
outputs destined for review). Topic names are lowercase, dot-separated, and
carry the owning domain as the first segment, e.g.
`trading_system.fills.v1`.

Use ZeroMQ when:

- the message is a **signal** or **tick** that must leave the sender as
  quickly as possible;
- the consumer is on the same host or a co-located host;
- replay is provided by upstream storage, not by the transport.

Streaming rules:

- **Backpressure**: producers must detect consumer lag and either drop
  (PUB/SUB market data) or slow down (Kafka OMS fills). Silent unbounded
  buffers are forbidden.
- **Schema evolution**: every topic carries a versioned protobuf schema in
  `shared_lib/contracts/`. Breaking changes require a new topic version.
- **Idempotency**: consumers treat at-least-once delivery as the default and
  deduplicate by `event_id`.
- **Tracing**: every event envelope carries `trace_id`, `span_id`, and
  `event_id` so OTel traces survive hops across transports.

## Schema and Contract Rules

- The source of truth for every cross-service payload is a schema under
  `shared_lib/contracts/`.
- protobuf is the canonical wire format for gRPC, ZeroMQ, and Kafka.
- pydantic is the canonical Python representation for REST payloads in the
  web_control_plane backend.
- Backwards-incompatible changes require a new contract version and a
  deprecation window recorded in the ADR that introduced the contract.
- No service may ship a custom wire format in place of the contracts
  library.

## Failure Modes

Every cross-service call must define:

- **timeout behaviour** — either a deadline (sync) or a retention/SLA
  (async);
- **retry behaviour** — explicit retry policy, including when NOT to retry
  (e.g. risk-engine rejections must not be retried);
- **dead-letter handling** — Kafka consumers route permanent failures to a
  DLQ topic named `<topic>.dlq.v1`;
- **partial-failure strategy** — responses document whether partial success
  is possible and how to reconcile;
- **circuit breakers** — services that call external providers wrap calls
  in circuit breakers to respect the multi-layer kill-switch principle.

## Security and Authentication

- All internal RPC traffic is authenticated. Default to mTLS between
  services; bearer-token auth is allowed only for browser ↔ backend paths.
- No service accepts unauthenticated mutating endpoints, including
  internal-only services.
- Private keys, broker credentials, and signer material never travel over
  application-level transports. They stay behind KMS/HSM/Vault as required
  by ADR-0006.
- OTel telemetry envelopes strip secrets and redact PII before export.

## Enforcement

- `ruff` rules forbid direct imports of raw HTTP libraries from domain
  packages; domain code must go through `shared_lib.transport`.
- CI contract tests verify every protobuf schema used on Kafka topics
  matches the registered version.
- The Phase 5 code_reviewer agent treats deviations from this document
  (missing idempotency keys, missing trace propagation, missing DLQ, etc.)
  as blocking review comments.
- Changes to this standard require an ADR update (ADR-0001 or a superseder)
  and a matching update to `tests/phase_0/test_service_communication_standards.py`.
