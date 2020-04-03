import numpy as np
import pandas as pd
import scipy.io as si

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