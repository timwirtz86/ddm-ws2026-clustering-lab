"""Helper library for the Data Driven Modelling clustering lab.

The library only does the "plumbing": loading the course datasets, saving
intermediate results between notebooks, running the small checks of the TODO
cells and drawing plots. Everything that is part of the learning goals
(preprocessing, distance functions, clustering algorithms, evaluation) is
implemented by you in the notebooks.
"""

from ddm.checks import check
from ddm.datasets import DATASETS, dataset_info, list_datasets, load_dataset, numeric_summary
from ddm.processed import load_array, load_json, processed_dir, save_array, save_json
from ddm.toy import load_toy_village
from ddm.viz import (
    plot_clusters_2d,
    plot_clusters_mds,
    plot_clusters_pca,
    plot_dendrogram,
    plot_toy_village,
)

__all__ = [
    "DATASETS",
    "check",
    "dataset_info",
    "list_datasets",
    "load_array",
    "load_dataset",
    "load_json",
    "load_toy_village",
    "numeric_summary",
    "plot_clusters_2d",
    "plot_clusters_mds",
    "plot_clusters_pca",
    "plot_dendrogram",
    "plot_toy_village",
    "processed_dir",
    "save_array",
    "save_json",
]
