import sys
import pandas 

nodes_path = sys.argv[1]
voc_path = sys.argv[2]

nodes = pandas.read_csv(nodes_path, sep=",")
voc = pandas.read_csv(voc_path, sep="\t")

nodes_map = dict(zip(nodes['id'].values, nodes['serialized_name'].values))

voc['Word'] = voc['Word'].apply(lambda node_id: nodes_map[node_id])

voc.to_csv("voc_fnames.tsv", sep="\t", index=False)