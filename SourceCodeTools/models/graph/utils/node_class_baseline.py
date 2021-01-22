import pandas
import sys
import os
import numpy as np

model_path = sys.argv[1]

nodes_path = os.path.join(model_path, "nodes.csv")

nodes = pandas.read_csv(nodes_path)

unique, indices, counts = np.unique(nodes['label'], return_inverse=True, return_counts=True)

p_class = counts / sum(counts)

labels = nodes['label'].values

random_baseline = sum(labels == np.random.choice(unique, labels.size)) / labels.size
random_baseline_weighted = sum(labels == np.random.choice(unique, labels.size, p=p_class)) / labels.size

print("Random baseline score: ", random_baseline)
print("Random baseline score with weighted classes: ", random_baseline_weighted)