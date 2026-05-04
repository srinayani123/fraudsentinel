"""Hybrid OOD query detector for the fraud knowledge base.

Two complementary signals + smart component selection:

  1. PCA reconstruction error — measures distance from the fraud-pattern subspace.
     Components selected via separability (t-statistic between ID and synthetic OOD
     projections) per Triantafyllopoulos et al. 2026, falling back to explained
     variance if no OOD reference set is provided.

  2. k-NN distance to ID set — measures distance to the closest k fraud patterns.
     Catches queries that are technically inside the subspace but far from any
     real pattern (sparse regions of ID space).

A query is flagged OOD only if BOTH signals agree. This reduces false positives
relative to either signal alone (Sun et al. 2022; Triantafyllopoulos et al. 2026).

References:
  - Triantafyllopoulos et al. (2026), "Knowing When Not to Answer: Lightweight
    KB-Aligned OOD Detection for Safe RAG" (arXiv:2508.02296)
  - Sun et al. (2022), "Out-of-distribution Detection with Deep Nearest Neighbors"
    (ICML 2022)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

OOD_FILE = Path("models/ood_pca_calibration.npz")
OOD_META = Path("models/ood_pca_meta.json")


class HybridOODDetector:
    """Hybrid OOD detector combining PCA reconstruction error + k-NN distance.

    PCA components are selected by separability when an OOD reference set is
    provided at fit time; otherwise they fall back to explained variance.
    """

    def __init__(
        self,
        components: np.ndarray,        # (n_components, embedding_dim)
        mean: np.ndarray,              # (embedding_dim,)
        id_embeddings: np.ndarray,     # (n_patterns, embedding_dim) - normalized
        pca_threshold: float,
        pca_borderline: float,
        knn_threshold: float,
        knn_borderline: float,
        n_components: int,
        k_neighbors: int,
        id_pca_p50: float,
        selection_method: str = "explained_variance",
    ):
        self.components = components
        self.mean = mean
        self.id_embeddings = id_embeddings
        self.pca_threshold = pca_threshold
        self.pca_borderline = pca_borderline
        self.knn_threshold = knn_threshold
        self.knn_borderline = knn_borderline
        self.n_components = n_components
        self.k_neighbors = k_neighbors
        self.id_pca_p50 = id_pca_p50
        self.selection_method = selection_method

    # ---------- Signal 1: PCA reconstruction error ----------
    def pca_error(self, embedding: np.ndarray) -> float:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        centered = embedding - self.mean
        projected = centered @ self.components.T
        reconstructed = projected @ self.components
        residual = centered - reconstructed
        return float(np.sum(residual ** 2))

    # ---------- Signal 2: k-NN distance to ID set ----------
    def knn_distance(self, embedding: np.ndarray) -> float:
        """Mean cosine distance to the k nearest fraud patterns."""
        if embedding.ndim == 2:
            embedding = embedding.reshape(-1)
        q = embedding / (np.linalg.norm(embedding) + 1e-9)
        cosines = self.id_embeddings @ q
        distances = 1.0 - cosines
        k = min(self.k_neighbors, len(distances))
        nearest = np.partition(distances, k - 1)[:k]
        return float(np.mean(nearest))

    # ---------- Combined verdict ----------
    def classify(self, embedding: np.ndarray) -> dict:
        """Classify query as 'in_domain', 'borderline', or 'out_of_domain'.

        Both signals must agree to flag OOD; both must agree on ID to pass.
        Disagreement → borderline (routes to LLM judge).
        """
        pca_err = self.pca_error(embedding)
        knn_dist = self.knn_distance(embedding)

        pca_verdict = self._verdict(pca_err, self.pca_borderline, self.pca_threshold)
        knn_verdict = self._verdict(knn_dist, self.knn_borderline, self.knn_threshold)

        if pca_verdict == "out_of_domain" and knn_verdict == "out_of_domain":
            verdict = "out_of_domain"
        elif pca_verdict == "in_domain" and knn_verdict == "in_domain":
            verdict = "in_domain"
        else:
            verdict = "borderline"

        pca_rel = self._relative(pca_err, self.id_pca_p50, self.pca_threshold)

        return {
            "verdict": verdict,
            "pca_error": pca_err,
            "pca_verdict": pca_verdict,
            "knn_distance": knn_dist,
            "knn_verdict": knn_verdict,
            "relative_score": pca_rel,
        }

    @staticmethod
    def _verdict(value: float, borderline: float, threshold: float) -> str:
        if value >= threshold:
            return "out_of_domain"
        elif value >= borderline:
            return "borderline"
        else:
            return "in_domain"

    @staticmethod
    def _relative(value: float, p50: float, threshold: float) -> float:
        denom = max(threshold - p50, 1e-9)
        rel = (value - p50) / denom
        return max(0.0, min(2.0, rel))

    # ----------------------------------------------------------------
    # Component selection methods
    # ----------------------------------------------------------------
    @staticmethod
    def _select_by_explained_variance(
        S: np.ndarray, Vt: np.ndarray, variance_target: float
    ) -> tuple[np.ndarray, int, str]:
        """Pick top components capturing variance_target fraction of total variance."""
        explained = (S ** 2)
        cum = np.cumsum(explained) / explained.sum()
        n = int(np.searchsorted(cum, variance_target) + 1)
        n = max(min(n, len(S)), 1)
        return Vt[:n], n, "explained_variance"

    @staticmethod
    def _select_by_separability(
        Vt: np.ndarray,
        id_centered: np.ndarray,
        ood_centered: np.ndarray,
        n_components: int,
    ) -> tuple[np.ndarray, int, str]:
        """Pick top components ranked by t-statistic of |projection| between ID and OOD.

        Per component v_i: project ID and OOD onto v_i, take absolute values, compute
        Welch's t-statistic. Larger |t| = better separation. Keep top-K.
        """
        # Project everything onto all candidate components
        id_proj = id_centered @ Vt.T   # (n_id, n_full)
        ood_proj = ood_centered @ Vt.T  # (n_ood, n_full)

        # Use absolute projections (we care about magnitude on each axis, not sign)
        id_abs = np.abs(id_proj)
        ood_abs = np.abs(ood_proj)

        # Welch's t-statistic per component
        id_mean = id_abs.mean(axis=0)
        ood_mean = ood_abs.mean(axis=0)
        id_var = id_abs.var(axis=0, ddof=1) + 1e-12
        ood_var = ood_abs.var(axis=0, ddof=1) + 1e-12
        denom = np.sqrt(id_var / len(id_abs) + ood_var / len(ood_abs))
        t_stats = np.abs(id_mean - ood_mean) / denom

        # Keep top-K components by |t|
        n = min(n_components, len(t_stats))
        top_indices = np.argsort(t_stats)[::-1][:n]
        # Sort indices ascending so the resulting basis is in original order (stable)
        top_indices = np.sort(top_indices)
        return Vt[top_indices], n, "separability"

    @classmethod
    def fit(
        cls,
        id_embeddings: np.ndarray,
        ood_reference_embeddings: Optional[np.ndarray] = None,
        variance_target: float = 0.80,
        n_components_separability: Optional[int] = None,
        pca_ood_percentile: float = 99.5,
        pca_borderline_percentile: float = 80.0,
        knn_ood_percentile: float = 99.5,
        knn_borderline_percentile: float = 80.0,
        k_neighbors: int = 5,
    ) -> "HybridOODDetector":
        """Fit both signals; calibrate thresholds against ID error distribution.

        If ood_reference_embeddings is provided, components are selected by
        separability (t-test between ID and OOD projections) — this is the
        2026 SOTA method from Triantafyllopoulos et al. Falls back to
        explained variance if no OOD reference set is given.
        """
        n_patterns, dim = id_embeddings.shape

        # Normalize ID embeddings
        norms = np.linalg.norm(id_embeddings, axis=1, keepdims=True)
        id_embeddings_normed = id_embeddings / (norms + 1e-9)

        # Center
        mean = id_embeddings_normed.mean(axis=0)
        id_centered = id_embeddings_normed - mean

        # Full SVD to get all candidate components
        _, S, Vt = np.linalg.svd(id_centered, full_matrices=False)

        # Pick components: separability if OOD reference set is provided, else EVR
        if ood_reference_embeddings is not None and len(ood_reference_embeddings) >= 10:
            ood_norms = np.linalg.norm(ood_reference_embeddings, axis=1, keepdims=True)
            ood_normed = ood_reference_embeddings / (ood_norms + 1e-9)
            ood_centered = ood_normed - mean

            # Decide how many components to keep
            n_target = n_components_separability
            if n_target is None:
                # Default: same count as EVR would have picked, so we're not
                # comparing apples to oranges later
                explained = (S ** 2)
                cum = np.cumsum(explained) / explained.sum()
                n_target = int(np.searchsorted(cum, variance_target) + 1)
                n_target = max(min(n_target, len(S)), 1)

            components, n_components, method = cls._select_by_separability(
                Vt, id_centered, ood_centered, n_target
            )
        else:
            components, n_components, method = cls._select_by_explained_variance(
                S, Vt, variance_target
            )

        # Compute PCA error distribution on ID set with selected components
        projected = id_centered @ components.T
        reconstructed = projected @ components
        residuals = id_centered - reconstructed
        pca_errors = np.sum(residuals ** 2, axis=1)
        pca_threshold = float(np.percentile(pca_errors, pca_ood_percentile))
        pca_borderline = float(np.percentile(pca_errors, pca_borderline_percentile))
        id_pca_p50 = float(np.percentile(pca_errors, 50))

        # Fit k-NN signal on ID set
        sims = id_embeddings_normed @ id_embeddings_normed.T
        np.fill_diagonal(sims, -np.inf)
        dists = 1.0 - sims
        np.fill_diagonal(dists, np.inf)
        k = min(k_neighbors, n_patterns - 1)
        knn_id_distances = np.array([
            np.mean(np.partition(dists[i], k - 1)[:k]) for i in range(n_patterns)
        ])
        knn_threshold = float(np.percentile(knn_id_distances, knn_ood_percentile))
        knn_borderline = float(np.percentile(knn_id_distances, knn_borderline_percentile))

        return cls(
            components=components,
            mean=mean,
            id_embeddings=id_embeddings_normed,
            pca_threshold=pca_threshold,
            pca_borderline=pca_borderline,
            knn_threshold=knn_threshold,
            knn_borderline=knn_borderline,
            n_components=n_components,
            k_neighbors=k,
            id_pca_p50=id_pca_p50,
            selection_method=method,
        )

    def save(self, npz_path: Path = OOD_FILE, meta_path: Path = OOD_META):
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            components=self.components,
            mean=self.mean,
            id_embeddings=self.id_embeddings,
        )
        meta = {
            "n_components": int(self.n_components),
            "embedding_dim": int(self.mean.shape[0]),
            "k_neighbors": int(self.k_neighbors),
            "pca_threshold": float(self.pca_threshold),
            "pca_borderline": float(self.pca_borderline),
            "knn_threshold": float(self.knn_threshold),
            "knn_borderline": float(self.knn_borderline),
            "id_pca_p50": float(self.id_pca_p50),
            "selection_method": str(self.selection_method),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls, npz_path: Path = OOD_FILE, meta_path: Path = OOD_META
    ) -> Optional["HybridOODDetector"]:
        if not npz_path.exists() or not meta_path.exists():
            return None
        try:
            data = np.load(npz_path)
            with open(meta_path) as f:
                meta = json.load(f)
            return cls(
                components=data["components"],
                mean=data["mean"],
                id_embeddings=data["id_embeddings"],
                pca_threshold=meta["pca_threshold"],
                pca_borderline=meta["pca_borderline"],
                knn_threshold=meta["knn_threshold"],
                knn_borderline=meta["knn_borderline"],
                n_components=meta["n_components"],
                k_neighbors=meta["k_neighbors"],
                id_pca_p50=meta["id_pca_p50"],
                selection_method=meta.get("selection_method", "explained_variance"),
            )
        except Exception:
            return None


# Backwards-compatible alias
PCAOODDetector = HybridOODDetector


def llm_judge_is_fraud_query(query: str) -> dict:
    """Fallback: ask Claude whether a query is plausibly about credit-card fraud."""
    import os

    import anthropic

    from src.utils.config import DEFAULT_ANTHROPIC_MODEL

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"is_fraud_related": True, "reasoning": "API key unavailable", "confidence": "low"}

    system_prompt = (
        "You are a fraud-domain query classifier. Given a query, decide if it is "
        "plausibly about credit card fraud, payment fraud, transaction risk, or "
        "card-not-present scams.\n\n"
        "Respond ONLY with JSON: "
        '{"is_fraud_related": true|false, "reasoning": "one sentence"}\n\n'
        "Examples:\n"
        '"small charges then large purchase" -> {"is_fraud_related": true, "reasoning": "matches card testing pattern"}\n'
        '"how do I bake a cake" -> {"is_fraud_related": false, "reasoning": "cooking question, not fraud"}\n'
        '"transactions from a new device at 3am" -> {"is_fraud_related": true, "reasoning": "device + timing anomaly"}'
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        resp = client.messages.create(
            model=model,
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        text = resp.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        result = json.loads(text)
        return {
            "is_fraud_related": bool(result.get("is_fraud_related", True)),
            "reasoning": str(result.get("reasoning", "")),
            "confidence": "high",
        }
    except Exception:
        return {"is_fraud_related": True, "reasoning": "judge unavailable", "confidence": "low"}
    