"""Repository and task locations independent of command entry-point paths."""
from pathlib import Path

TASK_ROOT = Path(__file__).resolve().parent
ROOT = TASK_ROOT.parents[5]
REFERENCE = TASK_ROOT / 'datasets/cmd_continue9D_flat.h5'
