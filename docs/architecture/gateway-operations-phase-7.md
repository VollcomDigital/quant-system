# Gateway Operations (Phase 7)

- Status: Accepted (Phase 7)
- Source ADRs: ADR-0005 (gateway architecture), ADR-0006 (signing /
  custody), ADR-0004 (control plane)
- Target phases: Phase 7 (this doc), Phase 9 deployment.

## Purpose

Phase 7 turns the abstract gateway scaffolding into operational rules
for the two paradigms ADR-0005 commits us to: TradFi (broker-shaped
order routing) and Web3 / DeFi (transaction construction + signing +
broadcast). This doc codifies the runbooks that the Phase 9
infrastructure deployment must implement.

## TradFi vs Web3 Gateway Split

Two paradigms with different state, security, and failure models:

- **TradFi** — `trading_system/gateways/tradfi/`. `AlpacaGateway` and
  `IBKRGateway` wrap a `BrokerClient` Protocol so vendor SDKs
  (alpaca-py, ib_insync) plug in cleanly. Order ids round-trip; the
  vendor never sees Phase 4 `OrderPayload` directly.
- **Web3 / DeFi** — `trading_system/gateways/web3/`. `Web3Gateway`
  builds `UnsignedTransaction` from ABI-driven calldata, simulates,
  signs (via KMS-backed `SigningClient`), and broadcasts. The gateway
  never holds private keys; signing is delegated. RPC access goes
  through Alchemy or Infura behind a `RpcClient` Protocol so vendor
  endpoints can be swapped without touching gateway logic.

`shared_gateways/` carries shared protocols (`Gateway`, `OrderAck`,
`GatewayOrder`) and the paper-trading `SimulatedGateway`. Both
TradFi and Web3 adapters produce the same `OrderAck` shape so the
EMS doesn't care about the underlying paradigm.

## IBKR Daily Restart Playbook

IBKR Gateway / TWS process *must* restart daily. The playbook:

1. **Pre-restart trading halt** (T-5 minutes):
   - `KillSwitch.trigger(reason="ibkr daily restart", actor="scheduler")`.
   - Web control plane displays the halt to operators (Phase 6 status
     endpoint).
2. **Cancel-all** (T-2 minutes):
   - `IBKRGateway.cancel_all()` → vendor `reqGlobalCancel`. The
     PanicPlaybook callback wraps this for the operator-triggered
     path.
3. **Container restart** (T-0):
   - The Phase 9 IB Gateway container (with IBC / IB Controller)
     restarts. `ib_insync` (or the cleaner native abstraction) is
     used in the Python control plane only — no vendor types leak
     into domain code.
4. **Re-authentication + reconnect** (T+30 seconds):
   - Container brings up a fresh session; `IBKRGateway` re-establishes
     the broker client connection.
5. **Reconciliation** (T+1 minute):
   - `OMS.reconcile(broker_positions=ibkr.positions())` returns a
     `ReconciliationDiff`. Empty diff → success → `KillSwitch.reset`
     (with an approval id from the scheduled-restart approval). Non-
     empty diff → operator review required, halt stays in place.

## Reconnect and Reconciliation

The same playbook applies to unscheduled disconnects: the
`HeartbeatTracker` in `shared_gateways.replay` flips the gateway to
`HealthStatus(ok=False)`, the risk monitor agent escalates, and the
operator triggers reconcile + reset through the Phase 6 control plane
endpoints.

Sequence-recovery harnesses live in `shared_gateways.replay`:
`replay_sequenced` refuses out-of-order or duplicate sequence numbers,
and `detect_gaps` returns the missing slots so the gateway can request
a re-send.

## DeFi Kill Controls

Three Layer-4 controls land in `trading_system/gateways/defi/`:

- `request_pause(target_contract)` — produces an `UnsignedTransaction`
  for a Pausable contract / Safe-module `pause()` call.
- `request_revoke_allowances(token, spenders)` — produces
  ERC-20 `approve(spender, 0)` requests.
- `ProtocolDenylist` — in-memory deny set the `Web3Gateway` consults
  before broadcasting; entries require an explicit `reason`.

Combined with Phase 6 `KillSwitch` and the Web3 `SigningClient`'s
role-checked `sign(...)` boundary, this gives us a multi-step DeFi
panic flow: deny the protocol → request allowance revocation → request
pool pause → broadcast through KMS-backed signer.

## No Browser Credentials

The browser **never holds** broker credentials, exchange credentials,
or private keys. KMS / Vault credentials live behind `SigningClient`
and `BrokerClient` implementations that the Phase 9 deployment wires
up. The web control plane only ever sees `OrderAck`, `HealthStatus`,
and `AuditEvent` payloads.

## Enforcement

- The Phase 7 invariant suite (`tests/phase_7/`) covers every
  contract above.
- The Phase 5 `code_reviewer` agent treats any PR that puts a
  vendor-specific SDK type into `web_control_plane.backend.*` as a
  blocking review finding.
- The Phase 6 "no submit_order endpoint" static test continues to
  hold: the only path into the gateway is via the OMS / EMS, never
  directly from the browser.
- Changes to this document require an ADR update (ADR-0005 or a
  superseder).
