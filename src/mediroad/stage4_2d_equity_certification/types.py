from __future__ import annotations
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any,Literal
import numpy as np
MetricName=Literal['min_sigungu_coverage','high_need_population']
@dataclass(frozen=True)
class MetricTarget:
    name:str; target:float; weights:np.ndarray; source:str
@dataclass
class CutRecord:
    name:str; indices:np.ndarray; values:np.ndarray; lower:float; upper:float; kind:str
    metric:str|None=None; anchor_size:int|None=None; base_value:float|None=None; target:float|None=None; source:str|None=None; metadata:dict[str,Any]=field(default_factory=dict)
@dataclass
class FixingRecord:
    candidate_index:int; venue_id:str; value:int; metric:str; target:float; optimistic_upper_bound:float; proof_type:str; metadata:dict[str,Any]=field(default_factory=dict)
@dataclass
class EvidenceSeed:
    source:str; selected_indices:np.ndarray; metrics:dict[str,float]; source_path:Path|None=None
@dataclass
class FrozenEquityContract:
    source_run_id:str|None; candidate_set:str; near_optimal_gap:float; efficiency_reference:float; total_population_floor:float; min_sigungu_reference:float; min_sigungu_retained_floor:float; high_need_reference:float; known_high_need_threshold:float; seeds:list[EvidenceSeed]; evidence:dict[str,Any]=field(default_factory=dict)
@dataclass
class SolveEvidence:
    backend:str; status:str; has_incumbent:bool; is_infeasible:bool; is_optimal:bool; objective_value:float|None; best_bound:float|None; relative_gap:float|None; wall_time_sec:float; node_count:int|None; selected_indices:np.ndarray|None; message:str; result_path:Path|None=None; metadata:dict[str,Any]=field(default_factory=dict)
@dataclass
class MetricCertificate:
    metric:MetricName; certified:bool; certificate:str; incumbent_value:float; best_bound:float|None; relative_gap:float|None; selected_indices:np.ndarray; retained_floor:float; rounds:int; evidence:list[dict[str,Any]]=field(default_factory=list)
