"""Portfolio optimizer.

Tiny inverse-variance fallback solver for `OptimizerRequest`. A real
convex solver (cvxpy, NVIDIA cuQuant, proprietary QP) plugs in behind
the same function signature. Phase 6 ships the contract + a reference
impl good enough for unit tests.
"""

from __future__ import annotations

from decimal import Decimal

from shared_lib.contracts import OptimizerRequest, OptimizerResponse

__all__ = ["solve_min_variance"]


def solve_min_variance(
    req: OptimizerRequest,
    *,
    covariance: dict[tuple[str, str], Decimal],
) -> OptimizerResponse:
    """Return inverse-variance weights, bounded to [0, 1] per symbol.

    This is the analytical min-variance solution when the covariance
    matrix is diagonal (or approximately so). Off-diagonal terms are
    ignored; the signature leaves them in place so a future solver
    upgrade does not break callers.
    """
    universe = req.universe
    variances: dict[str, Decimal] = {}
    for symbol in universe:
        key = (symbol, symbol)
        if key not in covariance:
            raise ValueError(
                f"covariance missing diagonal entry for {symbol!r}"
            )
        v = covariance[key]
        if v <= 0:
            raise ValueError(f"covariance[{symbol!r}] must be > 0")
        variances[symbol] = v

    # Inverse-variance weights; normalise to sum to 1.
    inv = {symbol: Decimal("1") / variances[symbol] for symbol in universe}
    total_inv = sum(inv.values(), start=Decimal("0"))
    weights = {symbol: inv[symbol] / total_inv for symbol in universe}

    # Apply per-symbol bounds if declared.
    for symbol, (lo, hi) in req.bounds.items():
        w = weights.get(symbol, Decimal("0"))
        weights[symbol] = max(lo, min(hi, w))

    # Renormalise after clamping to preserve sum=1.
    total = sum(weights.values(), start=Decimal("0"))
    if total == 0:
        raise ValueError("optimizer produced zero total weight after bounds clamp")
    weights = {k: v / total for k, v in weights.items()}

    objective_value = sum(
        (weights[s] * weights[s] * variances[s] for s in universe),
        start=Decimal("0"),
    )
    return OptimizerResponse(
        request_id=req.request_id,
        weights=weights,
        objective_value=objective_value,
    )
