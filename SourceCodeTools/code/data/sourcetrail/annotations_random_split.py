from SourceCodeTools.code.data.sourcetrail.Dataset import split, ensure_connectedness, ensure_valid_edges
import pandas as pd
import sys
from os.path import join, dirname

type_annotations_path = sys.argv[1]

type_ann = pd.read_csv(type_annotations_path).rename({
    "source_node_id": "dst",
    "target_node_id": "src"
}, axis=1)

train, test = split(type_ann, holdout_frac=0.4)

nodes = pd.DataFrame()

nodes['id'] = pd.concat([type_ann['src'], type_ann['dst']], axis=0).unique()

nodes, train = ensure_connectedness(nodes, train)

nodes, test = ensure_valid_edges(nodes, test, ignore_src=True)

train.rename({
    "dst":"source_node_id",
    "src":"target_node_id"
}, axis=1)[["id","type","source_node_id","target_node_id"]].to_csv(join(dirname(type_annotations_path), "annotations_train.csv"), index=False)
test.rename({
    "dst":"source_node_id",
    "src":"target_node_id"
}, axis=1)[["id","type","source_node_id","target_node_id"]].to_csv(join(dirname(type_annotations_path), "annotations_test.csv"), index=False)
