from __future__ import annotations

import json
import math
import pickle
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from .matrices import align_coverage, save_coverage_matrix
from .types import CoverageMatrix, Stage3Paths
from .utils import coalesce_column, read_table, sha256_file


SPEED_KMH = {
    "motorway": 90.0,
    "motorway_link": 55.0,
    "trunk": 75.0,
    "trunk_link": 50.0,
    "primary": 60.0,
    "primary_link": 45.0,
    "secondary": 50.0,
    "secondary_link": 40.0,
    "tertiary": 40.0,
    "tertiary_link": 35.0,
    "unclassified": 35.0,
    "residential": 30.0,
    "living_street": 15.0,
    "service": 20.0,
    "road": 30.0,
}


def _speed_from_tags(maxspeed: Any, highway: Any) -> float:
    if isinstance(maxspeed, (list, tuple, set, np.ndarray)):
        values = [_speed_from_tags(value, highway) for value in maxspeed]
        return min(values) if values else 30.0
    text = str(maxspeed or "").lower()
    nums = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if nums:
        speed = min(nums) * (1.609344 if "mph" in text else 1.0)
        return float(np.clip(speed, 5.0, 120.0))
    if isinstance(highway, (list, tuple, set, np.ndarray)):
        highway = next(iter(highway), "road")
    return float(SPEED_KMH.get(str(highway), 30.0))


def _core_vertex_ids(graph: Any) -> np.ndarray:
    attrs = set(graph.vs.attributes())
    for name in ["id", "osmid", "name"]:
        if name in attrs:
            return np.asarray([str(value) for value in graph.vs[name]], dtype=str)
    raise RuntimeError(f"Could not find OSM node ID in igraph vertex attributes: {sorted(attrs)}")


def _build_or_load_graph(
    package_root: Path,
    pbf_path: Path,
    cache_dir: Path,
    bbox_buffer: float,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    try:
        import geopandas as gpd
        from pyrosm import OSM
    except ImportError as exc:
        raise RuntimeError(
            "Network catchment validation requires geopandas, pyrosm and python-igraph. "
            "Install requirements_stage4.txt in the WSL Python 3.11 environment."
        ) from exc

    pbf_sha = sha256_file(pbf_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path = cache_dir / f"chungbuk_drive_core_{pbf_sha[:16]}.igraph.pickle"
    nodes_path = cache_dir / f"chungbuk_drive_core_nodes_{pbf_sha[:16]}.parquet"
    meta_path = cache_dir / f"chungbuk_drive_core_{pbf_sha[:16]}.json"
    if graph_path.exists() and nodes_path.exists() and meta_path.exists():
        import igraph as ig

        graph = ig.Graph.Read_Pickle(str(graph_path))
        nodes = pd.read_parquet(nodes_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return graph, nodes, meta

    boundary_path = package_root / "02_boundaries/chungbuk_admin_dong_2025Q2.gpkg"
    boundary = gpd.read_file(boundary_path).to_crs(4326)
    minx, miny, maxx, maxy = boundary.total_bounds
    bbox = [
        float(minx - bbox_buffer),
        float(miny - bbox_buffer),
        float(maxx + bbox_buffer),
        float(maxy + bbox_buffer),
    ]
    started = time.time()
    osm = OSM(str(pbf_path), bounding_box=bbox)
    nodes, edges = osm.get_network(nodes=True, network_type="driving")
    if nodes is None or edges is None or nodes.empty or edges.empty:
        raise RuntimeError("Pyrosm returned an empty driving network")
    if "length" not in edges.columns:
        edges_metric = edges.to_crs(5179)
        edges["length"] = edges_metric.geometry.length.to_numpy(dtype=float)
    maxspeed = edges["maxspeed"] if "maxspeed" in edges else pd.Series("", index=edges.index)
    highway = edges["highway"] if "highway" in edges else pd.Series("road", index=edges.index)
    edges["stage4_speed_kmh"] = [
        _speed_from_tags(speed, road) for speed, road in zip(maxspeed, highway)
    ]
    edges["travel_time_min"] = (
        pd.to_numeric(edges["length"], errors="coerce")
        / 1000.0
        / edges["stage4_speed_kmh"]
        * 60.0
    )
    edges = edges[np.isfinite(edges["travel_time_min"]) & edges["travel_time_min"].gt(0)].copy()

    graph = osm.to_graph(
        nodes,
        edges,
        graph_type="igraph",
        direction="oneway",
        network_type="driving",
        retain_all=True,
        simplify=False,
    )
    components = graph.connected_components(mode="STRONG")
    core = components.giant()
    core_ids = _core_vertex_ids(core)
    id_col = "id" if "id" in nodes.columns else nodes.index.name or "index"
    if id_col not in nodes.columns:
        nodes = nodes.reset_index().rename(columns={nodes.index.name or "index": id_col})
    nodes[id_col] = nodes[id_col].astype(str)
    core_nodes = nodes[nodes[id_col].isin(set(core_ids))][[id_col, "lon", "lat"]].copy()
    core_nodes = core_nodes.rename(columns={id_col: "node_id"})
    position = {value: idx for idx, value in enumerate(core_ids)}
    core_nodes["vertex_index"] = core_nodes["node_id"].map(position)
    if core_nodes["vertex_index"].isna().any() or len(core_nodes) != core.vcount():
        raise RuntimeError(
            f"Core graph node table mismatch: graph={core.vcount()}, table={len(core_nodes)}"
        )
    core_nodes["vertex_index"] = core_nodes["vertex_index"].astype(int)
    core_nodes = core_nodes.sort_values("vertex_index").reset_index(drop=True)

    core.write_pickle(str(graph_path))
    core_nodes.to_parquet(nodes_path, index=False)
    meta = {
        "pbf_path": str(pbf_path),
        "pbf_sha256": pbf_sha,
        "bbox_wgs84": bbox,
        "raw_nodes": int(len(nodes)),
        "raw_edges": int(len(edges)),
        "graph_vertices_all": int(graph.vcount()),
        "graph_edges_all": int(graph.ecount()),
        "core_vertices": int(core.vcount()),
        "core_edges": int(core.ecount()),
        "core_vertex_share": float(core.vcount() / max(1, graph.vcount())),
        "build_runtime_sec": float(time.time() - started),
        "definition": "Static OSM driving car-profile free-flow; oneway restrictions preserved; largest SCC core.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return core, core_nodes, meta


def _load_grid_geometry(package_root: Path, grid_ids: np.ndarray, aliases: dict[str, list[str]]) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("geopandas is required for network catchment validation") from exc
    candidates = list(
        (package_root / "06_population_grid").rglob("chungbuk_elderly_100m_calibrated_points_v6.gpkg")
    )
    if not candidates:
        candidates = list((package_root / "06_population_grid").rglob("*.gpkg"))
    if not candidates:
        raise FileNotFoundError("Could not locate the calibrated 100m elderly grid GPKG")
    gdf = gpd.read_file(candidates[0]).to_crs(4326)
    grid_col = coalesce_column(gdf, aliases["grid_id"])
    gdf[grid_col] = gdf[grid_col].astype(str)
    gdf = gdf.set_index(grid_col).reindex(np.asarray(grid_ids, dtype=str))
    if gdf.geometry.isna().any():
        missing = gdf[gdf.geometry.isna()].index.tolist()[:10]
        raise KeyError(f"Grid geometry missing for Stage 3 grid IDs, sample={missing}")
    return pd.DataFrame(
        {
            "grid_id": gdf.index.astype(str),
            "longitude": gdf.geometry.x.to_numpy(dtype=float),
            "latitude": gdf.geometry.y.to_numpy(dtype=float),
        }
    )


def _ensure_venue_coordinates(package_root: Path, candidates: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    out = candidates.copy()
    if {"longitude", "latitude"}.issubset(out.columns):
        lon = pd.to_numeric(out["longitude"], errors="coerce")
        lat = pd.to_numeric(out["latitude"], errors="coerce")
        if lon.notna().all() and lat.notna().all():
            out["longitude"] = lon
            out["latitude"] = lat
            return out
    canonical = package_root / "08_candidate_venues/candidate_venues_model_ready_final_v6.csv"
    source = pd.read_csv(canonical, low_memory=False)
    venue_col = coalesce_column(source, aliases["venue_id"])
    source[venue_col] = source[venue_col].astype(str)
    lon_col = coalesce_column(source, ["longitude", "lon", "x"])
    lat_col = coalesce_column(source, ["latitude", "lat", "y"])
    coords = source[[venue_col, lon_col, lat_col]].rename(
        columns={venue_col: "venue_id", lon_col: "longitude_source", lat_col: "latitude_source"}
    )
    out = out.merge(coords, on="venue_id", how="left", validate="one_to_one")
    out["longitude"] = pd.to_numeric(out.get("longitude"), errors="coerce").fillna(
        pd.to_numeric(out["longitude_source"], errors="coerce")
    )
    out["latitude"] = pd.to_numeric(out.get("latitude"), errors="coerce").fillna(
        pd.to_numeric(out["latitude_source"], errors="coerce")
    )
    if out[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Some Stage 4 candidates lack coordinates")
    return out.drop(columns=[c for c in ["longitude_source", "latitude_source"] if c in out])


def _snap_to_core(core_nodes: pd.DataFrame, longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    transformer = Transformer.from_crs(4326, 5179, always_xy=True)
    nx, ny = transformer.transform(
        core_nodes["lon"].to_numpy(dtype=float), core_nodes["lat"].to_numpy(dtype=float)
    )
    px, py = transformer.transform(np.asarray(longitude, dtype=float), np.asarray(latitude, dtype=float))
    distance, idx = cKDTree(np.c_[nx, ny]).query(np.c_[px, py])
    vertex = core_nodes.iloc[idx]["vertex_index"].to_numpy(dtype=np.int64)
    return vertex, np.asarray(distance, dtype=float)


def _binary_network_matrices(
    graph: Any,
    venue_vertices: np.ndarray,
    grid_vertices: np.ndarray,
    valid_venues: np.ndarray,
    valid_grids: np.ndarray,
    thresholds: list[int],
    batch_size: int,
) -> dict[int, sparse.csr_matrix]:
    rows: dict[int, list[np.ndarray]] = {threshold: [] for threshold in thresholds}
    cols: dict[int, list[np.ndarray]] = {threshold: [] for threshold in thresholds}

    valid_venue_rows = np.flatnonzero(np.asarray(valid_venues, dtype=bool))
    valid_grid_cols = np.flatnonzero(np.asarray(valid_grids, dtype=bool))
    unique_venue_vertices, venue_inverse = np.unique(
        np.asarray(venue_vertices, dtype=np.int64)[valid_venue_rows], return_inverse=True
    )
    unique_grid_vertices, grid_inverse = np.unique(
        np.asarray(grid_vertices, dtype=np.int64)[valid_grid_cols], return_inverse=True
    )

    # igraph 0.11 rejects duplicate target vertices. Deduplicating snapped
    # vertices also avoids recomputing identical shortest paths, then the
    # inverse arrays restore the original venue/grid universe exactly.
    for start in range(0, len(unique_venue_vertices), batch_size):
        stop = min(len(unique_venue_vertices), start + batch_size)
        batch_vertices = unique_venue_vertices[start:stop]
        distances = np.asarray(
            graph.distances(
                source=batch_vertices.tolist(),
                target=unique_grid_vertices.tolist(),
                weights="travel_time_min",
                mode="IN",
            ),
            dtype=float,
        )
        for local_row, unique_venue_row in enumerate(range(start, stop)):
            values = distances[local_row]
            original_venue_rows = valid_venue_rows[venue_inverse == unique_venue_row]
            for threshold in thresholds:
                reached_unique = values <= float(threshold)
                hit = valid_grid_cols[reached_unique[grid_inverse]]
                if not hit.size:
                    continue
                for original_venue_row in original_venue_rows:
                    rows[threshold].append(
                        np.full(hit.size, original_venue_row, dtype=np.int32)
                    )
                    cols[threshold].append(hit.astype(np.int32))
    output: dict[int, sparse.csr_matrix] = {}
    shape = (len(venue_vertices), len(grid_vertices))
    for threshold in thresholds:
        if rows[threshold]:
            r = np.concatenate(rows[threshold])
            c = np.concatenate(cols[threshold])
            data = np.ones(len(r), dtype=np.int8)
            output[threshold] = sparse.csr_matrix((data, (r, c)), shape=shape)
        else:
            output[threshold] = sparse.csr_matrix(shape, dtype=np.int8)
    return output


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def _topk_jaccard_by_admin(
    candidates: pd.DataFrame,
    ref_values: np.ndarray,
    alt_values: np.ndarray,
    k: int = 3,
) -> tuple[float, float]:
    temp = candidates[["venue_id", "admin_code"]].copy()
    temp["ref"] = ref_values
    temp["alt"] = alt_values
    jaccards: list[float] = []
    best_retention: list[float] = []
    for _, group in temp.groupby("admin_code"):
        ref_ids = set(group.nlargest(min(k, len(group)), "ref")["venue_id"])
        alt_ids = set(group.nlargest(min(k, len(group)), "alt")["venue_id"])
        union = ref_ids | alt_ids
        jaccards.append(len(ref_ids & alt_ids) / max(1, len(union)))
        ref_best = group.nlargest(1, "ref").iloc[0]["venue_id"]
        alt_best = group.nlargest(1, "alt").iloc[0]["venue_id"]
        best_retention.append(float(ref_best == alt_best))
    return float(np.mean(jaccards)), float(np.mean(best_retention))


def _admin_rank_stability(
    candidates: pd.DataFrame,
    ref_values: np.ndarray,
    alt_values: np.ndarray,
    *,
    top_k: int = 3,
) -> tuple[float, float]:
    temp = candidates[["admin_code"]].copy()
    temp["ref"] = np.asarray(ref_values, dtype=float)
    temp["alt"] = np.asarray(alt_values, dtype=float)
    correlations: list[float] = []
    informative = 0
    total = 0
    for _, group in temp.groupby("admin_code"):
        total += 1
        informative += int(len(group) > top_k)
        if len(group) < 2:
            continue
        rho = group["ref"].corr(group["alt"], method="spearman")
        if pd.notna(rho):
            correlations.append(float(rho))
    return (
        float(np.median(correlations)) if correlations else 0.0,
        float(informative / max(1, total)),
    )


def run_network_catchment_validation(
    package_root: Path,
    stage3: Stage3Paths,
    candidates: pd.DataFrame,
    grids: pd.DataFrame,
    euclidean_reference: CoverageMatrix,
    config: dict[str, Any],
    outdir: Path,
) -> dict[str, Any]:
    cfg = config["network_validation"]
    if not bool(cfg["enabled"]):
        return {"status": "SKIPPED_DISABLED", "matrices": {}, "summary": pd.DataFrame()}
    pbf_path = package_root / cfg["pbf_relative_path"]
    if not pbf_path.exists():
        raise FileNotFoundError(f"Network validation PBF not found: {pbf_path}")
    observed_pbf_sha = sha256_file(pbf_path)
    expected_pbf_sha = str(cfg.get("pbf_sha256", "")).strip().lower()
    if expected_pbf_sha and observed_pbf_sha != expected_pbf_sha:
        raise RuntimeError(
            f"Network validation PBF SHA mismatch: expected={expected_pbf_sha}, "
            f"actual={observed_pbf_sha}"
        )

    aliases = config["column_aliases"]
    candidates = _ensure_venue_coordinates(package_root, candidates, aliases)
    grid_geometry = _load_grid_geometry(package_root, grids["grid_id"].astype(str).to_numpy(), aliases)
    cache_dir = package_root / config["paths"]["stage4_root"] / "cache" / "road_graph"
    graph, core_nodes, graph_meta = _build_or_load_graph(
        package_root,
        pbf_path,
        cache_dir,
        float(cfg["graph_bbox_buffer_deg"]),
    )

    venue_vertices, venue_snap = _snap_to_core(
        core_nodes,
        candidates["longitude"].to_numpy(dtype=float),
        candidates["latitude"].to_numpy(dtype=float),
    )
    grid_vertices, grid_snap = _snap_to_core(
        core_nodes,
        grid_geometry["longitude"].to_numpy(dtype=float),
        grid_geometry["latitude"].to_numpy(dtype=float),
    )
    max_snap = float(cfg["max_snap_distance_m"])
    valid_venues = venue_snap <= max_snap
    valid_grids = grid_snap <= max_snap
    if valid_venues.mean() < 0.95 or valid_grids.mean() < 0.95:
        raise RuntimeError(
            f"Road core snap rate is too low: venues={valid_venues.mean():.3f}, grids={valid_grids.mean():.3f}"
        )

    thresholds = [int(value) for value in cfg["threshold_minutes"]]
    matrices = _binary_network_matrices(
        graph,
        venue_vertices,
        grid_vertices,
        valid_venues,
        valid_grids,
        thresholds,
        int(cfg["batch_size"]),
    )
    pop = pd.to_numeric(grids["elderly65_population"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    euclidean = align_coverage(
        euclidean_reference,
        candidates["venue_id"].astype(str).to_numpy(),
        grids["grid_id"].astype(str).to_numpy(),
        binary=True,
    )
    ref_exposure = np.asarray(euclidean @ pop).ravel()
    summary_rows: list[dict[str, Any]] = []
    matrix_objects: dict[str, CoverageMatrix] = {}
    matrix_dir = outdir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    for threshold, matrix in matrices.items():
        matrix_id = f"network_inbound_{threshold}min"
        alt_exposure = np.asarray(matrix @ pop).ravel()
        intersection = euclidean.multiply(matrix)
        inter_pop = np.asarray(intersection @ pop).ravel()
        union_pop = ref_exposure + alt_exposure - inter_pop
        weighted_jaccard = np.divide(
            inter_pop,
            union_pop,
            out=np.zeros_like(inter_pop, dtype=float),
            where=union_pop > 0,
        )
        top3, best = _topk_jaccard_by_admin(candidates, ref_exposure, alt_exposure, k=3)
        admin_rank_rho, top3_informative_fraction = _admin_rank_stability(
            candidates, ref_exposure, alt_exposure, top_k=3
        )
        row = {
            "matrix_id": matrix_id,
            "threshold_minutes": threshold,
            "spearman_vs_euclidean_5km": _spearman(ref_exposure, alt_exposure),
            "median_population_weighted_jaccard": float(np.median(weighted_jaccard)),
            "p25_population_weighted_jaccard": float(np.quantile(weighted_jaccard, 0.25)),
            "admin_top3_jaccard": top3,
            "admin_top3_informative_fraction": top3_informative_fraction,
            "admin_median_rank_spearman": admin_rank_rho,
            "admin_best_venue_retention": best,
            "median_exposure": float(np.median(alt_exposure)),
            "venue_snap_valid_rate": float(valid_venues.mean()),
            "grid_snap_valid_rate": float(valid_grids.mean()),
            "nnz": int(matrix.nnz),
        }
        row["comparison_score"] = (
            0.35 * max(0.0, row["spearman_vs_euclidean_5km"])
            + 0.35 * max(0.0, row["admin_median_rank_spearman"])
            + 0.30 * row["median_population_weighted_jaccard"]
        )
        summary_rows.append(row)
        meta = save_coverage_matrix(
            matrix,
            candidates["venue_id"].astype(str).to_numpy(),
            grids["grid_id"].astype(str).to_numpy(),
            matrix_dir,
            matrix_id,
            {
                "method": "OSM_static_free_flow_inbound",
                "threshold_minutes": threshold,
                "pbf_sha256": sha256_file(pbf_path),
                "definition": "Grid-to-venue inbound static OSM car-profile free-flow; not observed travel or live traffic.",
            },
        )
        matrix_objects[matrix_id] = CoverageMatrix(
            matrix_id=matrix_id,
            matrix=matrix,
            venue_ids=candidates["venue_id"].astype(str).to_numpy(),
            grid_ids=grids["grid_id"].astype(str).to_numpy(),
            method="OSM_static_free_flow_inbound",
            radius_or_scale=float(threshold),
            source_path=matrix_dir / f"{matrix_id}.npz",
            metadata=meta,
        )

    summary = pd.DataFrame(summary_rows).sort_values("comparison_score", ascending=False).reset_index(drop=True)
    best_matrix_id = str(summary.iloc[0]["matrix_id"]) if len(summary) else None
    venue_detail = candidates[["venue_id", "admin_code", "sigungu"]].copy()
    venue_detail["osm_core_snap_distance_m"] = venue_snap
    venue_detail["osm_core_snap_valid"] = valid_venues
    grid_snap_summary = {
        "count": int(len(grid_snap)),
        "valid": int(valid_grids.sum()),
        "valid_rate": float(valid_grids.mean()),
        "median_m": float(np.median(grid_snap)),
        "p99_m": float(np.quantile(grid_snap, 0.99)),
        "max_m": float(np.max(grid_snap)),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "network_catchment_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    venue_detail.to_csv(outdir / "network_venue_snap_quality.csv", index=False, encoding="utf-8-sig")
    payload = {
        "status": "PASS",
        "best_network_matrix_id": best_matrix_id,
        "network_matrices": list(matrix_objects),
        "graph": graph_meta,
        "venue_snap": {
            "count": int(len(venue_snap)),
            "valid": int(valid_venues.sum()),
            "valid_rate": float(valid_venues.mean()),
            "median_m": float(np.median(venue_snap)),
            "p99_m": float(np.quantile(venue_snap, 0.99)),
            "max_m": float(np.max(venue_snap)),
        },
        "grid_snap": grid_snap_summary,
        "interpretation": (
            "Network catchments are static inbound OSM car-profile free-flow proxies. "
            "They are robustness comparators, not observed elderly travel or live traffic."
        ),
    }
    (outdir / "NETWORK_CATCHMENT_VALIDATION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        **payload,
        "matrices": matrix_objects,
        "summary": summary,
        "candidate_table": candidates,
    }
