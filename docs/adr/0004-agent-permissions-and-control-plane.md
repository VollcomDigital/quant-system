# ADR 0004: Agent Permissions and Control Plane

- Status: Proposed
- Owners: AI Agents, Platform, Risk
- Target phase: Phase 5

## Context

Agents are intended to operate as scoped programmatic workers, not unrestricted
LLM wrappers. They need controlled access to research code, backtests,
observability data, and reporting channels without being able to bypass risk
controls or mutate production-critical systems directly.

## Decision Drivers

- least-privilege access
- human-approval boundaries
- auditable tool use
- deterministic control-plane interfaces
- safe CI and production integration

## Options to Evaluate

1. Tool-based control plane with explicit allowlists
2. Direct file-system and API access per agent role
3. Queue-based job execution with mediated side effects

## Proposed Direction

Use a mediated control plane where agents operate through explicit tools and
service APIs. Agent permissions are scoped by role:

- researcher agents can propose code and trigger controlled backtests
- review agents can analyze diffs and CI artifacts
- risk agents can interpret telemetry and recommend interventions

Any action with production impact should require either a deterministic policy
pass or a human approval step.

## Questions to Resolve

- Which actions may be executed automatically versus approval-gated?
- What is the audit-log schema for agent tool calls?
- Which notification sinks are mandatory for high-severity agent findings?

## Exit Criteria

- role matrix approved
- control-plane interfaces identified
- approval boundaries documented
