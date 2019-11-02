import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import normalize

allowed_list = "/Volumes/External/dev/code-embedding/voc.tsv"
in_vects = "/Volumes/External/dev/code-embedding/in_m.txt"

lda = dict()
with open("allowed_parent.txt") as lda_functions:

    vectors = np.loadtxt("allowed_parents_vec.txt", delimiter="\t")

    for ind, line in enumerate(lda_functions):
        func = line.strip()[:-2]
        if func:
            # print(func)
            lda[func] = vectors[ind, :]

emb = dict()
with open(allowed_list) as al:
    vectors = np.loadtxt(in_vects, delimiter="\t")

    for ind, line in enumerate(al):
        func = line.strip()
        if func:
            func = func.split("\t")[0]
            # print(func)
            if func in lda:
                emb[func] = vectors[ind-1, :]


order = list(emb.keys())
print(len(lda), len(emb))
print(len(order))

lda_emb = np.zeros((len(order), 300))
emb_emb = np.zeros((len(order), 100)) 



for ind, e in enumerate(order):
    lda_emb[ind, :] = lda[e]
    emb_emb[ind, :] = emb[e]

lda_emb = normalize(lda_emb, axis=1)
emb_emb = normalize(emb_emb, axis=1)

queries_ind = np.random.randint(0, len(order), 7000)
# print(queries)

lda_queries = lda_emb[queries_ind, :]
emb_queries = emb_emb[queries_ind, :]

# lda_similarity = np.argsort(lda_emb @ lda_queries.T, axis=0)
# emb_similarity = np.argsort(emb_emb @ emb_queries.T, axis=0)

lda_similarity_map = np.argsort(lda_emb @ lda_queries.T, axis=0)
# emb_similarity_map = np.argsort(emb_emb @ emb_queries.T, axis=0)
emb_similarity_map = lda_similarity_map[-10:, :]
print("Considering top: %d, query sample size: %d" % emb_similarity_map.shape)
# print(emb_similarity_map)

lda_similarity = lda_emb @ lda_queries.T
emb_similarity = emb_emb @ emb_queries.T

lda_map = np.zeros(emb_similarity_map.shape)
emb_map = np.zeros(emb_similarity_map.shape)
for j in range(emb_similarity_map.shape[1]):
    col = np.sort(emb_similarity_map[:, j])
    lda_map[:, j] = lda_similarity[col, j]
    emb_map[:, j] = emb_similarity[col, j]

lda_similarity = np.argsort(lda_map, axis=0)
emb_similarity = np.argsort(emb_map, axis=0)

# print(lda_similarity)
# print(emb_similarity)


# lda_similarity = np.argsort((lda_emb @ lda_queries.T)[emb_similarity], axis=0)
# emb_similarity = np.argsort((emb_emb @ emb_queries.T)[emb_similarity], axis=0)
# print(lda_sim[lda_similarity[-10:, 0],0])

rho_acc = 0
uncorr = 0; corr = 0
corr_rho = 0
uncorr_rho = 0
for ind, q in enumerate(queries_ind):
    rho, p = spearmanr(lda_similarity[:, ind], emb_similarity[:, ind])
    rho_acc += rho
    if p < 0.01:
        corr += 1
        corr_rho += rho
    else:
        uncorr += 1
        uncorr_rho += rho

    print("uncorr: %d, corr: %d" % (uncorr, corr), end="\r")
    # if p < 0.05:
    #     print("%s\t%.4f\t%.4f" % (order[q], rho, p))
print()
print("uncorr_rho: %.4f, corr_rho: %.4f" % (uncorr_rho/(uncorr + 1), corr_rho/(corr+1)))
rho_acc /= queries_ind.size
print(rho_acc)
