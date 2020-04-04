import numpy as np
import pandas as pd
import scipy.io as si
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from random import random
import joblib
from coclust.clustering.spherical_kmeans import SphericalKmeans
from Cluster_Ensembles import CSPA, HGPA, MCLA
from cluster_ensembles_sample import  cluster_ensembles

def sparsity(mat):
    return np.round(1 - (np.count_nonzero(mat) / float(mat.size)), 3)

def balance(class_):
    element_level = []
    for i in np.unique(class_):
        element_level.append(list(class_).count(i))
    min_ = min(element_level)
    max_ = max(element_level)
    return np.round(min_/max_, 3)

def create_data_table(data: {}):
    
    nbr_doc = []
    nbr_terms = []
    sparsities = []
    cl = []
    blces = []
    dataset_name = [*data.keys()]
    
    for name,d in data.items():
        nbr_doc.append(d[0].shape[0])
        nbr_terms.append(d[0].shape[1])
        sparsities.append(sparsity(d[0]))
        cl.append(len(np.unique(d[1])))
        blces.append(balance(d[1]))
        
    table_dict = {"dataset": dataset_name, "# document": nbr_doc, "# mots": nbr_terms,"# class": cl, "sparsité": sparsities, "pt class/ gr class": blces  }
    table_frame = pd.DataFrame(table_dict)
    return table_frame

def load_data(data_names,dir_path='.'):
    data = {}
    for db in data_names:
        temp = si.loadmat(f'{dir_path}/{db}.mat')
        data[db] = [temp['mat'].toarray(), temp['labels'][0], temp['fea'], temp['label_names'] ]
    return data


def find_partitions(data, algo,minimized,method="CI", n=5,  runs=15):

    results = []
    for _ in range(runs):
        copy_algo = algo
        copy_algo.fit(data)
        results.append(copy_algo)

    # joblib.dump(results, f'./{name}.joblib')

    if method == "CI":
        vals = [part.criterion for part in results]
    elif method == "CM":
        vals = [part.modularity for part in results]
    else:
        vals = [random() for _ in results]

    if minimized:
        ind = np.array(vals).argsort()[:n]
    else:
        ind = np.array(vals).argsort()[-n:]
    
    best_part = np.take(results, ind)
    row_labels = np.array([item.row_labels_ for item in best_part])
    
    return row_labels

def co_association(label):
    
    return (label == label[:, np.newaxis]) * 1

def total_association(labels):
            
    return np.sum( [co_association(label) for label in labels], axis=0)



def find_asso_nmi_ari(algo, pred_lab, reel_lab,):

    association = total_association(pred_lab)
    algo.fit(association)
    asso_result = [NMI(reel_lab,algo.row_labels_), ARI(reel_lab, algo.rolabels_)]
    
    return asso_result

def evalute_(label_true,pred_labels):
    
    nmi_ = {}
    ari_ = {}
    
    for i,label in enumerate(pred_labels):
        nmi_[f'best{i+1}'] = NMI(label_true,label)
        ari_[f'best{i+1}'] = ARI(label_true,label)
    return nmi_, ari_

def sk_best_partitions(data,k,n=5,runs=20):
    
    partitions = []
    
    for i in range(runs):
        sk = SphericalKmeans(k)
        sk.fit(data)
        partitions.append(sk)
        
    best_index = [ part.criterion for part in partitions]

    ind = np.array(best_index).argsort()[-n:]
    best_part = np.take(partitions, ind)
    
    row_lab = np.array([item.labels_ for item in best_part])
    
    return best_part, row_lab

### DEPREDICATE

# def coc_best_partitions(data,n_row,n_col,n=5, runs=20):
    
#     partitions = []
    
#     for i in range(runs):
#         coc = CoclustInfo(n_row_clusters=n_row, n_col_clusters=n_col)
#         coc.fit(data)
#         partitions.append(coc)
        
#     best_index = [ part.criterion for part in partitions]

#     ind = np.array(best_index).argsort()[:n]
#     best_part = np.take(partitions, ind)
    
#     row_lab = np.array([item.row_labels_ for item in best_part])
    
#     return best_part, row_lab
