import dgl
import pandas as pd
import argparse
import ast
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from SourceCodeTools.code.data.sourcetrail.file_utils import read_processed_bodies, unpersist
from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import AstProcessor, standardize_new_edges, \
    ReplacementNodeResolver, MentionTokenizer
from SourceCodeTools.code.data.sourcetrail.sourcetrail_compute_function_diameter import compute_diameter
from SourceCodeTools.code.data.sourcetrail.sourcetrail_parse_bodies2 import has_valid_syntax
from SourceCodeTools.code.python_ast import AstGraphGenerator
from SourceCodeTools.nlp import create_tokenizer


def process_code(source_file_content, node_resolver, mention_tokenizer):
    ast_processor = AstProcessor(source_file_content)
    try: # TODO recursion error does not appear consistently. The issue is probably with library versions...
        edges = ast_processor.get_edges(as_dataframe=False)
    except RecursionError:
        return None

    if len(edges) == 0:
        return None

    edges = standardize_new_edges(edges, node_resolver, mention_tokenizer)

    return edges


def compute_gnn_passings(body, mention_tokenizer):

    node_resolver = ReplacementNodeResolver()

    source_file_content = body.lstrip()



    edges = process_code(
        source_file_content, node_resolver, mention_tokenizer
    )

    if edges is None:
        return None

    edges = pd.DataFrame(edges).rename({"src": "source_node_id", "dst": "target_node_id"}, axis=1)
    diameter = compute_diameter(edges, func_id=0)

    G = dgl.DGLGraph()
    G.add_edges(edges["source_node_id"], edges["target_node_id"])

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(diameter)

    loader = dgl.dataloading.NodeDataLoader(
        G, G.nodes(), sampler, batch_size=1, shuffle=True, num_workers=0)
    for input_nodes, seeds, blocks in loader:
        num_edges = 0
        for block in blocks:
            num_edges += block.num_edges()
        if num_edges != 0:
            break

    return num_edges

def compute_transformer_passings(body, bpe):
    num_tokens = len(bpe(body))
    return num_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bodies")
    parser.add_argument("bpe_path")
    parser.add_argument("--num_layers", default=8, type=int)

    args = parser.parse_args()

    bodies = unpersist(args.bodies)
    bpe = create_tokenizer(type="bpe", bpe_path=args.bpe_path)
    mention_tokenizer = MentionTokenizer(args.bpe_path, create_subword_instances=True, connect_subwords=False)

    lengths_tr = {}
    lengths_gnn = {}
    ratio = []

    for body in tqdm(bodies["body"]):
        if not has_valid_syntax(body):
            continue

        n_tokens = compute_transformer_passings(body, bpe)
        n_edges = compute_gnn_passings(body, mention_tokenizer)

        if n_tokens not in lengths_tr:
            lengths_tr[n_tokens] = []
        if n_tokens not in lengths_gnn:
            lengths_gnn[n_tokens] = []

        lengths_tr[n_tokens].append(n_tokens ** 2 * args.num_layers)
        lengths_gnn[n_tokens].append(n_edges)# * args.num_layers)
        ratio.append((n_tokens, n_edges))

    for key in lengths_tr:
        data_tr = np.array(lengths_tr[key])
        data_gnn = np.array(lengths_gnn[key])

        lengths_tr[key] = np.mean(data_tr)#, np.std(data_tr))
        lengths_gnn[key] = np.mean(data_gnn)#, np.std(data_gnn))

    data_ratios = np.array(ratio)

    plt.plot(data_ratios[:, 0], data_ratios[:, 1], "*")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Edges")
    plt.savefig("tokens_edges.png")
    plt.close()

    plt.hist(data_ratios[:, 1] / data_ratios[:, 0], bins=20)
    plt.xlabel("Number of edges / Number of tokens")
    plt.savefig("ratio.png")
    plt.close()

    ratio = data_ratios[:, 1] / data_ratios[:, 0]
    ratio = (np.mean(ratio), np.std(ratio))

    plt.plot(list(lengths_tr.keys()), np.log10(np.array(list(lengths_tr.values()))), "*")
    plt.plot(list(lengths_gnn.keys()), np.log10(np.array(list(lengths_gnn.values()))), "*")
    plt.plot(list(lengths_gnn.keys()), np.log10(np.array(list(lengths_gnn.values())) * args.num_layers), "*")
    plt.legend([f"Transformer {args.num_layers} layers", "GNN L layers", f"GNN L*{args.num_layers} layers"])
    plt.xlabel("Number of Tokens")
    plt.ylabel("log10(Number of Message Exchanges)")
    plt.savefig("avg_passings.png")
    plt.close()





if __name__ == "__main__":
    main()