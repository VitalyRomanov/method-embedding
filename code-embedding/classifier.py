# import tensorflow as tf
import numpy as np
import argparse
import pandas 
import sklearn
from pprint import pprint
from functools import reduce

#%%

parser = argparse.ArgumentParser(description='Train skipgram embeddings')
parser.add_argument('--in', dest='in_m', type=str, default='in_m.txt',
                    help='')
parser.add_argument('--out', dest='out_m', type=str, default='out_m.txt',
                    help='')
parser.add_argument('--voc', dest='voc', type=str, default='voc_fnames.tsv',
                    help='Path to vocabulary')
parser.add_argument('--allowed', dest='allowed', type=str, default='reused_call_nodes.txt',
                    help='Path to vocabulary')

args = parser.parse_args()

#%%

print("Loading embeddings...", end = "")
in_matr = np.loadtxt(args.in_m)
out_matr = np.loadtxt(args.out_m)
print("done")

#%%

def get_allowed_name_ids(voc_path, allowed_path):
    all_fnames = pandas.read_csv(voc_path, sep="\t")['Word'].values.tolist()

    allowed_fnames = open(allowed_path, "r").read().strip().split("\n")

    all_fnames_map = dict(zip(all_fnames, range(len(all_fnames))))
    ids = [all_fnames_map[name] for name in allowed_fnames if all_fnames_map.get(name, -1) != -1]
    return list(zip(allowed_fnames, ids))

allowed_name_id = get_allowed_name_ids(args.voc, args.allowed)

#%%

allowed_fnames, ids = zip(*allowed_name_id)
# min_module_depth = reduce(
#     lambda x, y: min(x,y),
#     map(lambda name: len(name.split(".")), allowed_fnames)
# )

ids = np.array(ids, dtype=np.int32)

reprs_in = in_matr[ids, :]
reprs_out = out_matr[ids, :]

fnames = list(map(lambda name: name.split(".")[-1], allowed_fnames))
mnames = list(map(lambda name: name.split(".")[0], allowed_fnames))

#%% 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report

def evaluate_model(model, X, y):
    return pandas.DataFrame(classification_report(y,
                                              model.predict(X),
                                              output_dict=True)).iloc[:-1, :2]

def evaluate(data, labels):

    param_grid = {
        'hidden_layer_sizes': [(70,), (50,), (70, 100,), (50, 70,),],
        'activation': ['relu'],
        'solver': ['adam'],
        'learning_rate_init': [0.001],
        'early_stopping': [True]
    }

    X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size = 0.7, random_state = 42)

    for params in ParameterGrid(param_grid):
        model = MLPClassifier()

        model.fit(X_train, y_train)

        print(params)
        print(evaluate_model(model, X_test, y_test))
        break

print("\n\n\n reps=in, names=func")
evaluate(reprs_in, fnames)
print("\n\n\n reps=in, names=func")
evaluate(reprs_out, fnames)
print("\n\n\n reps=in, names=func")
evaluate(reprs_in, mnames)
print("\n\n\n reps=in, names=func")
evaluate(reprs_out, mnames)
        

