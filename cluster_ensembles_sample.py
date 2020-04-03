import numpy as np
import Cluster_Ensembles as CE
from Cluster_Ensembles.Cluster_Ensembles import (
    build_hypergraph_adjacency,
    store_hypergraph_adjacency,
    ceEvalMutual
)
from Cluster_Ensembles import CSPA, HGPA, MCLA
import tables


def cluster_ensembles(cluster_runs, verbose, N_clusters_max, method):
    hdf5_file_name = 'tmp_graph'
    fileh = tables.open_file(hdf5_file_name, 'w')
    fileh.create_group(fileh.root, 'consensus_group')
    fileh.close()
    hypergraph_adjacency = build_hypergraph_adjacency(cluster_runs)
    store_hypergraph_adjacency(hypergraph_adjacency, hdf5_file_name)
    cluster_ensemble = method(
        hdf5_file_name, cluster_runs, verbose, N_clusters_max)
    score = ceEvalMutual(cluster_runs, cluster_ensemble, verbose)
    return score, cluster_ensemble


clustering_1 = [0, 1, 1, 2, 0, 2, 1, 0, 2, 1]
clustering_2 = [0, 1, 2, 0, 0, 2, 2, 1, 2, 1]
clustering_3 = [2, 0, 0, 2, 1, 1, 1, 0, 1, 2]
cluster_runs = np.array([clustering_1, clustering_2, clustering_3])
print("cluster_runs", cluster_runs)


score, consensus_clustering_labels = cluster_ensembles(
    cluster_runs, verbose=True, N_clusters_max=3, method=CSPA)
print('score', score)
# consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 3)
print("consensus_clustering_labels", consensus_clustering_labels)


score, consensus_clustering_labels = cluster_ensembles(
    cluster_runs, verbose=True, N_clusters_max=3, method=MCLA)
print('score', score)
# consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 3)
print("consensus_clustering_labels", consensus_clustering_labels)


score, consensus_clustering_labels = cluster_ensembles(
    cluster_runs, verbose=True, N_clusters_max=3, method=HGPA)
print('score', score)
# consensus_clustering_labels = CE.cluster_ensembles(cluster_runs, verbose = True, N_clusters_max = 3)
print("consensus_clustering_labels", consensus_clustering_labels)
