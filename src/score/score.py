"""Scoring utilities for PVX-like candidate selection.

All masks are expected to be 2D arrays of shape (256,256) with values in {0,1}.
Scores are intentionally simple and robust.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np


def _as_u8(mask: np.ndarray) -> np.ndarray:
    if mask is None:
        raise ValueError("mask is None")
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {m.shape}")
    return (m > 0).astype(np.uint8)


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: int
    coverage: float
    inside_boundary_violation: int
    keepout_violation: int
    compactness_proxy: int | None
    final_score: float


def score_candidate(
    mask_u8: np.ndarray,
    boundary_mask_u8: np.ndarray,
    keepout_mask_u8: np.ndarray,
    *,
    compute_compactness: bool = True,
) -> Dict[str, Any]:
    """Score a single candidate mask against boundary and keepouts.

    final_score = coverage - 5*(keepout_violation>0) - 5*(inside_boundary_violation>0)

    Returns a plain dict for easy JSON/table output.
    """

    m = _as_u8(mask_u8)
    b = _as_u8(boundary_mask_u8)
    k = _as_u8(keepout_mask_u8)

    boundary_sum = float(b.sum())
    coverage = float(m.sum()) / (boundary_sum + 1e-6)

    outside = (1 - b).astype(np.uint8)
    inside_boundary_violation = int((m & outside).sum())
    keepout_violation = int((m & k).sum())

    compactness_proxy = None
    if compute_compactness:
        try:
            import cv2

            # connectedComponents counts background as 0
            num_labels, _labels = cv2.connectedComponents(m)
            compactness_proxy = int(max(0, int(num_labels) - 1))
        except Exception:
            compactness_proxy = None

    final_score = float(coverage)
    final_score -= 5.0 * (1.0 if keepout_violation > 0 else 0.0)
    final_score -= 5.0 * (1.0 if inside_boundary_violation > 0 else 0.0)

    return {
        "coverage": float(coverage),
        "inside_boundary_violation": int(inside_boundary_violation),
        "keepout_violation": int(keepout_violation),
        "compactness_proxy": (None if compactness_proxy is None else int(compactness_proxy)),
        "final_score": float(final_score),
    }


def rank_candidates(
    candidates: np.ndarray,
    boundary_mask_u8: np.ndarray,
    keepout_mask_u8: np.ndarray,
    *,
    compute_compactness: bool = True,
) -> List[Dict[str, Any]]:
    """Rank candidate masks by final_score (descending)."""

    if candidates is None:
        return []
    arr = np.asarray(candidates)
    if arr.ndim != 3:
        raise ValueError(f"candidates must be (N,H,W), got shape {arr.shape}")

    b = _as_u8(boundary_mask_u8)
    k = _as_u8(keepout_mask_u8)

    scores: List[CandidateScore] = []
    for i in range(int(arr.shape[0])):
        m = (arr[i] > 0).astype(np.uint8)
        metrics = score_candidate(m, b, k, compute_compactness=compute_compactness)
        scores.append(
            CandidateScore(
                candidate_id=int(i),
                coverage=float(metrics["coverage"]),
                inside_boundary_violation=int(metrics["inside_boundary_violation"]),
                keepout_violation=int(metrics["keepout_violation"]),
                compactness_proxy=(None if metrics["compactness_proxy"] is None else int(metrics["compactness_proxy"])),
                final_score=float(metrics["final_score"]),
            )
        )

    scores.sort(key=lambda s: (s.final_score, s.coverage), reverse=True)
    return [asdict(s) for s in scores]
