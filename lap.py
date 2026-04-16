"""Small compatibility shim for Ultralytics tracker assignment on platforms without native `lap` wheels."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "0.5.12"


def lapjv(cost_matrix, extend_cost: bool = True, cost_limit: float | None = None):
    """Approximate the `lap.lapjv` interface using SciPy's Hungarian solver."""

    matrix = np.asarray(cost_matrix, dtype=float)
    num_rows, num_cols = matrix.shape
    row_assignments = np.full(num_rows, -1, dtype=int)
    col_assignments = np.full(num_cols, -1, dtype=int)

    if matrix.size == 0:
        return 0.0, row_assignments, col_assignments

    row_ind, col_ind = linear_sum_assignment(matrix)

    total_cost = 0.0
    for row_idx, col_idx in zip(row_ind, col_ind):
        value = float(matrix[row_idx, col_idx])
        if cost_limit is not None and value > cost_limit:
            continue
        row_assignments[row_idx] = col_idx
        col_assignments[col_idx] = row_idx
        total_cost += value

    return total_cost, row_assignments, col_assignments
