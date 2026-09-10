from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .formulation import CertificationFormulation
from .types import Floor, LinearMipModel, Seed


@dataclass
class SwapSearchResult:
    selected_indices: np.ndarray
    objective_value: float
    rounds: int
    backend: str


def _read_venue_ids(path: Path) -> list[str]:
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    col = next((c for c in ["venue_id", "resolved_venue_id", "physical_venue_id"] if c in frame), None)
    if col is None:
        return []
    return frame[col].dropna().astype(str).tolist()


def _map_ids(formulation: CertificationFormulation, ids: Iterable[str]) -> np.ndarray | None:
    pos = {value: i for i, value in enumerate(formulation.candidates["venue_id"].astype(str))}
    aliases = getattr(formulation, "venue_alias_map", {})
    mapped: list[int] = []
    for value in ids:
        source = str(value)
        resolved = str(aliases.get(source, source))
        if resolved not in pos:
            return None
        mapped.append(pos[resolved])
    selected = np.asarray(mapped, dtype=int)
    if len(set(map(int, selected))) != len(selected):
        return None
    return selected


def discover_seed_selections(
    project_root: Path,
    formulation: CertificationFormulation,
    stage41: Any,
    *,
    scenario: str,
    candidate_set: str,
    greedy_indices: np.ndarray | None = None,
    previous_stage_indices: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    seeds: list[tuple[str, np.ndarray]] = []
    if previous_stage_indices is not None:
        seeds.append(("PREVIOUS_LEXICOGRAPHIC_STAGE", np.asarray(previous_stage_indices, dtype=int)))
    if greedy_indices is not None:
        seeds.append(("CONSTRAINED_GREEDY", np.asarray(greedy_indices, dtype=int)))

    stage41_paths = {
        "efficiency": getattr(stage41, "plan_efficiency", None),
        "balanced": getattr(stage41, "plan_balanced", None),
        "equity": getattr(stage41, "plan_equity", None),
    }
    primary = stage41_paths.get(scenario)
    if primary:
        selected = _map_ids(formulation, _read_venue_ids(Path(primary)))
        if selected is not None:
            seeds.append((f"STAGE4_1_{scenario.upper()}", selected))
    for name, path in stage41_paths.items():
        if name == scenario or path is None:
            continue
        selected = _map_ids(formulation, _read_venue_ids(Path(path)))
        if selected is not None:
            seeds.append((f"STAGE4_1_{name.upper()}", selected))

    patterns = [
        project_root
        / "outputs/model_v1/10_stage4_2/runs/**/candidate_sets"
        / candidate_set
        / f"plan__{scenario}.csv",
        project_root
        / "outputs/model_v1/10_stage4_2_diagnostic/runs/**/candidate_sets"
        / candidate_set
        / f"plan__{scenario}.csv",
        project_root
        / "outputs/model_v1/10_stage4_2_certification/runs/**/candidate_sets"
        / candidate_set
        / f"plan__{scenario}.csv",
    ]
    for pattern in patterns:
        for raw in sorted(glob.glob(str(pattern), recursive=True), reverse=True)[:20]:
            path = Path(raw)
            selected = _map_ids(formulation, _read_venue_ids(path))
            if selected is not None:
                seeds.append((f"PRIOR_RUN::{path.parent.parent.parent.name}", selected))

    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for source, selected in seeds:
        key = tuple(sorted(map(int, selected)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((source, selected))
    return unique


def evaluate_seed_pool(
    formulation: CertificationFormulation,
    model: LinearMipModel,
    seed_selections: list[tuple[str, np.ndarray]],
) -> list[Seed]:
    output: list[Seed] = []
    for source, selected in seed_selections:
        valid, reason = formulation.validate_selection(selected)
        if not valid:
            output.append(Seed(source=source, selected_indices=selected, rejection_reason=reason))
            continue
        full = formulation.complete_solution(model, selected)
        violations = formulation.feasibility_violations(model, full)
        feasible = bool(violations["feasible"])
        objective = formulation.objective_for_selection(model.objective_name, selected)
        output.append(
            Seed(
                source=source,
                selected_indices=np.asarray(selected, dtype=int),
                full_values=full,
                feasible=feasible,
                objective_value=float(objective),
                metrics=formulation.metrics(selected),
                rejection_reason=None if feasible else f"model violations={violations}",
            )
        )
    return output


def choose_seed(seeds: list[Seed], sense: str) -> Seed | None:
    feasible = [seed for seed in seeds if seed.feasible and seed.objective_value is not None]
    if not feasible:
        return None
    reverse = sense == "max"
    return sorted(feasible, key=lambda s: float(s.objective_value), reverse=reverse)[0]


class SingleSwapImprover:
    """Vectorized one-swap incumbent improvement with optional CUDA evaluation.

    CUDA is only used for coverage metric evaluation. The final MIP certificate is
    still produced by HiGHS on the CPU.
    """

    def __init__(self, formulation: CertificationFormulation, *, allow_gpu: bool = True) -> None:
        self.f = formulation
        self.backend = "NUMPY"
        self.xp = np
        self.B = self.f._candidate_pattern.toarray().astype(np.int8, copy=False)
        self.pop = self.f.population
        self.need = self.f.need_weighted
        self.high = self.f.high_need
        if allow_gpu:
            try:
                import cupy as cp  # type: ignore

                # Trigger a tiny allocation so missing CUDA runtime/driver fails here.
                cp.zeros(1, dtype=cp.int8)
                self.xp = cp
                self.B = cp.asarray(self.B)
                self.pop = cp.asarray(self.pop)
                self.need = cp.asarray(self.need)
                self.high = cp.asarray(self.high)
                self.backend = "CUPY_CUDA"
            except Exception:
                self.xp = np
                self.backend = "NUMPY"

    def _to_numpy(self, value: Any) -> np.ndarray:
        if self.backend == "CUPY_CUDA":
            return self.xp.asnumpy(value)
        return np.asarray(value)

    def _hard_mask(self, selected: np.ndarray, out_idx: int) -> np.ndarray:
        n = self.f.nx
        mask = np.ones(n, dtype=bool)
        remaining = [int(i) for i in selected if int(i) != int(out_idx)]
        mask[remaining] = False
        frame = self.f.candidates
        admin_counts = frame.iloc[remaining]["admin_code"].astype(str).value_counts()
        sig_counts = frame.iloc[remaining]["sigungu"].astype(str).value_counts()
        clusters = set(
            frame.iloc[remaining]["cluster_id"].fillna(frame.iloc[remaining]["venue_id"]).astype(str)
        )
        for i, row in frame.iterrows():
            if not mask[i]:
                continue
            if admin_counts.get(str(row["admin_code"]), 0) >= self.f.max_admin:
                mask[i] = False
                continue
            if sig_counts.get(str(row["sigungu"]), 0) >= self.f.max_sigungu:
                mask[i] = False
                continue
            raw_cluster = row.get("cluster_id", row["venue_id"])
            cluster = str(row["venue_id"]) if pd.isna(raw_cluster) else str(raw_cluster)
            if self.f.one_cluster and cluster in clusters:
                mask[i] = False
        return mask

    def _coverage_metrics_for_all_in(self, cover_count_minus_out: Any) -> dict[str, np.ndarray]:
        xp = self.xp
        covered = (cover_count_minus_out[None, :] + self.B) > 0
        total = self._to_numpy(covered @ self.pop).astype(float)
        need = self._to_numpy(covered @ self.need).astype(float)
        high = self._to_numpy(covered @ self.high).astype(float)
        min_ratio = np.full(self.f.nx, np.inf, dtype=float)
        for sigungu, idx_np in self.f.sigungu_pattern_indices.items():
            total_sig = self.f.sigungu_population_total[sigungu]
            if total_sig <= 0:
                continue
            idx = xp.asarray(idx_np) if self.backend == "CUPY_CUDA" else idx_np
            ratio = self._to_numpy(covered[:, idx] @ self.pop[idx]).astype(float) / total_sig
            min_ratio = np.minimum(min_ratio, ratio)
        return {
            "total_population": total,
            "need_weighted": need,
            "high_need_population": high,
            "min_sigungu_coverage": min_ratio,
        }

    def _cost_for_all_in(self, selected: np.ndarray, out_idx: int) -> np.ndarray:
        base_indices = [int(i) for i in selected if int(i) != int(out_idx)]
        candidate_base = float(self.f._candidate_cost[base_indices].sum()) + self.f._candidate_cost
        frame = self.f.candidates
        counts = frame.iloc[base_indices]["sigungu"].astype(str).value_counts().reindex(self.f.sigungu_values, fill_value=0)
        count_base = counts.to_numpy(int)
        penalty = float(
            self.f.base_config.get("optimization", {}).get("visit_location_deviation_penalty", 100.0)
        )
        target = self.f.visit_count / max(1, self.f.ns)
        output = np.empty(self.f.nx, dtype=float)
        sig_pos = {s: i for i, s in enumerate(self.f.sigungu_values)}
        for i, sigungu in enumerate(self.f.candidate_sigungu):
            c = count_base.copy()
            c[sig_pos[str(sigungu)]] += 1
            output[i] = candidate_base[i] + np.abs(c - target).sum() * penalty
        return output

    @staticmethod
    def _satisfies_floors(metrics: dict[str, np.ndarray], floors: list[Floor]) -> np.ndarray:
        n = len(next(iter(metrics.values())))
        mask = np.ones(n, dtype=bool)
        for floor in floors:
            values = metrics[floor.objective]
            if floor.sense == "max":
                mask &= values >= floor.value - 1e-7 * max(1.0, abs(floor.value))
            else:
                mask &= values <= floor.value + 1e-7 * max(1.0, abs(floor.value))
        return mask

    def improve(
        self,
        selected_indices: np.ndarray,
        *,
        objective_name: str,
        sense: str,
        floors: list[Floor],
        max_rounds: int = 25,
    ) -> SwapSearchResult:
        selected = np.asarray(selected_indices, dtype=int).copy()
        current = self.f.objective_for_selection(objective_name, selected)
        rounds = 0
        xp = self.xp
        B_selected = self.B[xp.asarray(selected) if self.backend == "CUPY_CUDA" else selected]
        cover_count = B_selected.sum(axis=0)

        for _ in range(max_rounds):
            best_value = current
            best_swap: tuple[int, int] | None = None
            for out_idx in selected.tolist():
                base_count = cover_count - self.B[int(out_idx)]
                metrics = self._coverage_metrics_for_all_in(base_count)
                metrics["cost"] = self._cost_for_all_in(selected, int(out_idx))
                valid = self._hard_mask(selected, int(out_idx))
                valid &= self._satisfies_floors(metrics, floors)
                valid[int(out_idx)] = True
                values = metrics[objective_name]
                if sense == "max":
                    values = np.where(valid, values, -np.inf)
                    in_idx = int(np.argmax(values))
                    candidate_value = float(values[in_idx])
                    improved = candidate_value > best_value + 1e-8 * max(1.0, abs(best_value))
                else:
                    values = np.where(valid, values, np.inf)
                    in_idx = int(np.argmin(values))
                    candidate_value = float(values[in_idx])
                    improved = candidate_value < best_value - 1e-8 * max(1.0, abs(best_value))
                if improved:
                    best_value = candidate_value
                    best_swap = (int(out_idx), in_idx)
            if best_swap is None:
                break
            out_idx, in_idx = best_swap
            position = int(np.flatnonzero(selected == out_idx)[0])
            selected[position] = in_idx
            cover_count = cover_count - self.B[out_idx] + self.B[in_idx]
            current = best_value
            rounds += 1
        return SwapSearchResult(
            selected_indices=np.asarray(selected, dtype=int),
            objective_value=float(current),
            rounds=rounds,
            backend=self.backend,
        )
