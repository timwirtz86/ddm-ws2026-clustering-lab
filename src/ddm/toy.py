"""The warm-up toy example: buildings on a village map.

This is the same data that is printed on the warm-up handout. Each building has
a position on the map (x, y in km), a type, a size and an occupancy status. The
map also has a river that runs roughly along x = 5.

The data is hand-made so that several "reasonable" clusterings exist:

* by position: a west and an east village, a small group at the river, a row of
  farms along the south-east road, one isolated farm;
* by position *and* the river: the river group falls apart into two halves;
* by building type, by size or by occupancy: groups that cut across the map.

Which of these is "right" depends on who looks at the map and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# id, x, y, type, size, occupied
_BUILDINGS = [
    # West village
    ("A", 1.8, 7.9, "apartment", "large", True),
    ("B", 2.4, 8.5, "apartment", "large", True),
    ("C", 2.9, 7.6, "house", "small", True),
    ("D", 2.2, 7.0, "house", "small", False),
    ("E", 3.3, 8.2, "apartment", "small", True),
    ("F", 3.0, 6.6, "farm", "large", True),
    ("G", 1.5, 6.9, "house", "small", True),
    # East village
    ("H", 7.1, 7.4, "house", "small", True),
    ("I", 7.7, 8.1, "house", "large", True),
    ("J", 8.3, 7.5, "apartment", "large", True),
    ("K", 7.5, 6.7, "house", "small", False),
    ("L", 8.6, 8.4, "apartment", "small", True),
    ("M", 8.9, 6.9, "farm", "small", True),
    # Group at the river (west bank: N, O; east bank: P, Q)
    ("N", 4.15, 4.25, "house", "small", True),
    ("O", 4.30, 3.60, "house", "large", True),
    ("P", 5.45, 4.10, "house", "small", False),
    ("Q", 5.60, 3.45, "house", "large", True),
    # Farms along the south-east road
    ("R", 6.2, 0.8, "farm", "large", True),
    ("S", 6.9, 1.2, "farm", "small", True),
    ("T", 7.6, 1.5, "farm", "large", False),
    ("U", 8.3, 1.9, "farm", "small", True),
    ("V", 8.9, 2.4, "farm", "large", True),
    ("W", 9.4, 2.9, "farm", "small", True),
    # Isolated farm in the south-west
    ("X", 1.0, 1.3, "farm", "large", True),
]


def river_x(y: np.ndarray | float) -> np.ndarray | float:
    """x-coordinate of the river centre line at height y."""
    return 5.0 + 0.3 * np.sin(y)


def road_xy() -> tuple[np.ndarray, np.ndarray]:
    """Polyline of the south-east road (passes the farms R to W)."""
    x = np.array([5.6, 6.2, 6.9, 7.6, 8.3, 8.9, 9.4, 10.0])
    y = np.array([0.45, 0.65, 1.0, 1.3, 1.7, 2.2, 2.7, 3.2])
    return x, y


def load_toy_village() -> pd.DataFrame:
    """Return the 24 buildings of the warm-up map as a DataFrame.

    Columns: ``id``, ``x``, ``y`` (km), ``type`` (house/apartment/farm),
    ``size`` (small/large), ``occupied`` (bool) and ``bank`` (west/east of
    the river).
    """
    df = pd.DataFrame(_BUILDINGS, columns=["id", "x", "y", "type", "size", "occupied"])
    df["bank"] = np.where(df["x"] < river_x(df["y"]), "west", "east")
    return df.set_index("id")
