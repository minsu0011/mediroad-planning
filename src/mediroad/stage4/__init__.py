"""Stage 4 provisional multi-objective spatial optimization.

This package treats Stage 1 Need, Stage 2A Specialty Gap, and Stage 3 spatial
coverage as separate evidence layers. It does not predict patient counts and it
does not assign exact dates.
"""

from .finalizer import run_stage4_finalization
from .pipeline import run_stage4

__all__ = ["run_stage4", "run_stage4_finalization"]
