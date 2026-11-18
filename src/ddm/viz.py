"""Plotting helpers: show clusterings in 2D.

Real datasets have more than two attributes, so we cannot draw them directly.
The helpers project the data onto two dimensions and colour the points by
cluster label. Comparing several clusterings side by side is the main use:
pass a dict ``{"name": labels, ...}`` and get one panel per clustering.

Label ``-1`` is drawn in grey and treated as "noise" (DBSCAN convention).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

_PALETTE = [
    "#2a6fdb",
    "#e8743b",
    "#19a979",
    "#c81e0f",
    "#945ecf",
    "#13a4b4",
    "#bf399e",
    "#6c8893",
    "#ee6868",
    "#525df4",
]
_NOISE = "#b0b0b0"


def _as_labelings(labelings) -> dict[str, np.ndarray]:
    if isinstance(labelings, Mapping):
        return {str(k): np.asarray(v) for k, v in labelings.items()}
    return {"clustering": np.asarray(labelings)}


def _is_noise(label) -> bool:
    return isinstance(label, (int, np.integer)) and label == -1


def _sorted_labels(labels: np.ndarray) -> list:
    """Unique labels in a stable order, so that equal labels get equal colours."""
    unique = list(pd.unique(labels))
    try:
        return sorted(unique)
    except TypeError:  # mixed types: keep the order of appearance
        return unique


def _scatter_panels(
    coords: np.ndarray,
    labelings: dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    suptitle: str | None,
    point_names: Sequence[str] | None,
):
    n = len(labelings)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.4 * nrows), squeeze=False)
    for ax, (name, labels) in zip(axes.flat, labelings.items()):
        if len(labels) != len(coords):
            raise ValueError(
                f"'{name}': {len(labels)} labels for {len(coords)} points."
            )
        clusters = [lab for lab in _sorted_labels(labels) if not _is_noise(lab)]
        if len(clusters) < len(pd.unique(labels)):
            clusters.append(-1)  # noise last, in grey
        for i, lab in enumerate(clusters):
            mask = labels == lab
            is_noise = _is_noise(lab)
            color = _NOISE if is_noise else _PALETTE[i % len(_PALETTE)]
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=22,
                c=color,
                label="noise" if is_noise else str(lab),
                alpha=0.85,
                edgecolors="white",
                linewidths=0.4,
            )
        if point_names is not None:
            for (px, py), txt in zip(coords, point_names):
                ax.annotate(str(txt), (px, py), fontsize=7, xytext=(3, 3), textcoords="offset points")
        n_clusters = len([lab for lab in pd.unique(labels) if not _is_noise(lab)])
        ax.set_title(f"{name}  ({n_clusters} clusters)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if n_clusters <= 10:
            ax.legend(fontsize=8, loc="best", frameon=False)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def plot_clusters_pca(
    X,
    labelings,
    *,
    title: str | None = None,
    point_names: Sequence[str] | None = None,
):
    """Project a numeric feature matrix onto its first two principal components.

    Parameters
    ----------
    X : array-like of shape (n_objects, n_features)
        The *preprocessed* numeric features (scaled, encoded). PCA is
        sensitive to scale, so pass the same matrix you cluster on.
    labelings : array of labels, or dict ``{"name": labels}``
        One panel is drawn per labeling, all with the same projection.
    title : optional figure title
    point_names : optional names written next to each point (small data only)

    Returns
    -------
    matplotlib.figure.Figure
    """
    X = np.asarray(X, dtype=float)
    if np.isnan(X).any():
        raise ValueError("X contains NaN. Handle missing values before plotting.")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_ * 100
    return _scatter_panels(
        coords,
        _as_labelings(labelings),
        f"PC 1 ({ev[0]:.0f} % of variance)",
        f"PC 2 ({ev[1]:.0f} % of variance)",
        title,
        point_names,
    )


def plot_clusters_mds(
    D,
    labelings,
    *,
    title: str | None = None,
    point_names: Sequence[str] | None = None,
    random_state: int = 0,
):
    """Embed a precomputed distance matrix in 2D (metric MDS) and plot clusterings.

    Use this when you cluster on your own distance matrix (e.g. Gower) and have
    no numeric feature matrix for PCA. MDS places the points so that their 2D
    distances approximate the given distances as well as possible.
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square distance matrix.")
    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="classical_mds",
        n_init=1,
        random_state=random_state,
    )
    coords = mds.fit_transform(D)
    return _scatter_panels(
        coords, _as_labelings(labelings), "MDS 1", "MDS 2", title, point_names
    )


def plot_dendrogram(
    Z,
    *,
    labels: Sequence[str] | None = None,
    cut_height: float | None = None,
    title: str | None = None,
    ax=None,
    **kwargs,
):
    """Draw a dendrogram from a linkage matrix in SciPy format.

    ``Z`` has ``n - 1`` rows ``[cluster_a, cluster_b, distance, size]``.
    Clusters ``0 .. n-1`` are the original objects; the cluster created in
    merge step ``k`` gets the id ``n + k``. Your own agglomerative
    implementation can return exactly this format.

    Extra keyword arguments go to ``scipy.cluster.hierarchy.dendrogram``, e.g.
    ``truncate_mode="lastp", p=30`` to show only the last 30 merges of a
    large tree.
    """
    Z = np.asarray(Z, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4.5))
    kwargs.setdefault("leaf_font_size", 8)
    dendrogram(Z, labels=labels, ax=ax, color_threshold=cut_height, **kwargs)
    if cut_height is not None:
        ax.axhline(cut_height, color="#c81e0f", linestyle="--", linewidth=1)
    ax.set_ylabel("merge distance")
    if title:
        ax.set_title(title)
    return ax.figure


def plot_clusters_2d(
    coords,
    labelings,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str | None = None,
    point_names: Sequence[str] | None = None,
    equal_aspect: bool = False,
):
    """Plot clusterings of data that already *is* two-dimensional (no projection)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_objects, 2).")
    fig = _scatter_panels(coords, _as_labelings(labelings), xlabel, ylabel, title, point_names)
    if equal_aspect:
        for ax in fig.axes:
            ax.set_aspect("equal", adjustable="box")
    return fig


def plot_toy_village(labelings, *, title: str | None = None):
    """Draw clusterings of the warm-up village map, with river and road.

    ``labelings`` is one array of 24 labels (in the order of
    ``load_toy_village()``) or a dict ``{"name": labels}``.
    """
    from ddm.toy import load_toy_village, river_x, road_xy

    village = load_toy_village()
    coords = village[["x", "y"]].to_numpy()
    fig = _scatter_panels(
        coords, _as_labelings(labelings), "x (km)", "y (km)", title, list(village.index)
    )
    y_line = np.linspace(0, 10, 200)
    road_x, road_y = road_xy()
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        ax.plot(river_x(y_line), y_line, color="#7fb3e6", linewidth=4, alpha=0.5, zorder=0)
        ax.plot(road_x, road_y, color="#999999", linewidth=2, linestyle=":", zorder=0)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal", adjustable="box")
    return fig
