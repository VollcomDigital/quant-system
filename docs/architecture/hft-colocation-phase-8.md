# HFT Co-Location and Bare-Metal Deployment (Phase 8)

- Status: Accepted (Phase 8)
- Source ADRs: ADR-0003 (two-speed runtime), ADR-0005 (gateway
  architecture)
- Applies to: every production deployment of
  `trading_system/native/hft_engine/*`.

## Purpose

The HFT path cannot hit its latency budgets on shared cloud compute.
Phase 8 codifies the deployment environment Phase 9 infrastructure
must provide: co-located bare-metal hosts, kernel-bypass NICs, tuned
kernels, and PTP-synchronised clocks. Mid-frequency workloads
continue to live on Kubernetes; the HFT engine does not.

## Co-Location Requirements

- The HFT engine runs on bare-metal hosts **co-located** inside the
  exchange data centre or in a carrier-neutral facility with a
  cross-connect into the exchange match engine.
- Each venue has at least one primary + one hot-spare host. The
  hot-spare runs the same native binary on the same NIC type, ready
  for manual failover.
- Cross-connects are documented in the Phase 9 IaC tree with explicit
  cage + rack + port identifiers.

## Bare-Metal Hardware Baseline

- CPU: a single-socket server with a recent generation of x86_64
  (AMD EPYC or Intel Xeon with high clock speed); hyper-threading
  disabled.
- RAM: ECC DDR5, enough for 2 GB **hugepages** per engine + headroom.
- Storage: NVMe for local journals only; historical datasets live on
  the Phase 2 Parquet store, not on the HFT host.
- BIOS: C-states disabled, Turbo Boost pinned to base clock, SpeedStep
  disabled, power profile set to maximum performance, IOMMU enabled
  for DPDK / vfio-pci.

## Network Fabric

- Kernel-bypass NIC (Solarflare onload, Mellanox ConnectX with DPDK
  or RDMA, or equivalent). The NIC is bound exclusively to the HFT
  engine process; no kernel stack traffic on the same port.
- PTP (Precision Time Protocol) synchronisation on every HFT host;
  deviation > 1 microsecond is a `HealthStatus(ok=False)` incident.
- Tx/Rx IRQ affinity pinned to the same NUMA node as the process
  running `hft_engine.core`.

## Kernel and OS Tuning

- Linux kernel with the `isolcpus` boot parameter covering the cores
  assigned to `hft_engine.core`. Kernel task scheduling stays off
  those cores.
- CPU pinning is mandatory: the `core` event loop, the `network` Rx
  handler, and the `fast_inference` worker each get a dedicated core
  on the same NUMA node.
- Hugepages pre-allocated at boot (1 GB / 2 GB pages). Normal
  `malloc` on the critical path is forbidden.
- IOMMU enabled and configured for vfio-pci passthrough when DPDK is
  used.
- Kernel samepage merging (KSM), swap, Transparent Hugepages
  (coalescer-side), and cgroup memory pressure notifications all
  disabled on the HFT partition.

## Deployment Boundary vs Cloud Runtime

- Mid-frequency services (OMS / EMS / web control plane / model
  serving) run on Kubernetes per Phase 9 IaC.
- The HFT engine **is not** deployed on Kubernetes. It is **not** in
  the Phase 9 Kubernetes overlays. It ships as a bare-metal binary +
  systemd unit managed by the Phase 9 `infrastructure/bare_metal/`
  playbooks.
- Secrets + credentials reach the HFT host through KMS-issued
  short-lived tokens (ADR-0006); they never traverse the cloud
  control plane.
- **No Python** process runs on the HFT host's isolated cores. The
  only Python allowed on a bare-metal HFT machine is an offline
  provisioning helper that exits before the HFT process starts.

## Enforcement

- Phase 9 infrastructure must supply separate playbooks for
  bare-metal HFT vs Kubernetes mid-frequency; the Phase 5
  `code_reviewer` agent flags any PR that adds an HFT workload to
  the Kubernetes overlays.
- The Phase 8 `HFTModelCard` + `LatencyBudget` gates refuse live
  deployment without a measured co-located benchmark run.
- Changes to this document require an ADR update (ADR-0003 or a
  superseder).
