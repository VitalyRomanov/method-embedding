allowed_list = "/Volumes/External/dev/code-embedding/voc.tsv"
import numpy as np

allowed = set(line.split("\t")[0] for line in open(allowed_list) if len(line.split())> 1)

parent_list = open("compacted_for_desc.txt").readlines()
vectors = np.loadtxt("doc_topic_matrix_300_compacted.txt", delimiter="\t")

allowed_name = []
allowed_vec = []

# print(allowed)

for ind, name in enumerate(parent_list):
    name = name.strip()

    if name[:-2] in allowed:
        allowed_name.append(name)
        allowed_vec.append(vectors[ind, :])

allowed_vec = np.array(allowed_vec)
np.savetxt("allowed_parents_vec.txt", allowed_vec, delimiter="\t")
with open("allowed_parent.txt", "w") as al:
    for a in allowed_name:
        al.write("%s\n" % a)


