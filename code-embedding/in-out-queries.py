from sklearn.neighbors import BallTree
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

allowed_list = "voc.tsv"
in_vects = "in_m.txt"
out_vects = "out_m.txt"

f_names = []

with open(allowed_list) as al:
    for ind, line in enumerate(al):
        if ind==0: continue
        func = line.strip()
        if func:
            func = func.split("\t")[0]
            f_names.append(func)

in_vectors = np.loadtxt(in_vects, delimiter="\t")
out_vectors = np.loadtxt(out_vects, delimiter="\t")
# in_vectors -= in_vectors.mean(axis=0)
# out_vectors -= out_vectors.mean(axis=0)
# in_vectors = normalize(in_vectors, axis=1)
# out_vectors = normalize(out_vectors, axis=1)

# in_tree = BallTree(in_vectors)
# out_tree = BallTree(out_vectors)

# while True:
#     f_n = input("Enter function name: ")
#     f_n = f_n.strip()
#     if f_n in f_names:
#         f_v = in_vectors[f_names.index(f_n), :].reshape(1, -1)
#         print(f_v.shape)
#         dist, ind = out_tree.query(f_v, 10)
#         # for d, i in zip(dist[0], ind[0]):
#         #     print("%s\t%.4f" % (f_names[i], d))
#         # print(dist, ind)
#         for i in ind[0]:
#             print("%s\t%.4f" % (f_names[i], cosine_distances(in_vectors[f_names.index(f_n)].reshape(1, -1), out_vectors[i].reshape(1, -1))))

while True:
    f_n = input("Enter function name: ")
    f_n = f_n.strip()
    if f_n in f_names:
        f_v = in_vectors[f_names.index(f_n), :]
        score = out_vectors @ f_v
        ind = np.flip(np.argsort(score))[:20]
        # for d, i in zip(dist[0], ind[0]):
        #     print("%s\t%.4f" % (f_names[i], d))
        # print(dist, ind)
        for i in ind:
            print("%s\t%.4f" % (f_names[i], in_vectors[f_names.index(f_n)].dot(out_vectors[i])))
    else:
        print("Nothing found")