import logging
from collections import defaultdict
from os.path import join, dirname

import dgl
import pandas as pd
import argparse
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from SourceCodeTools.code.ast import has_valid_syntax
from SourceCodeTools.code.data.file_utils import unpersist
from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import AstProcessor, standardize_new_edges, \
    ReplacementNodeResolver, MentionTokenizer
from SourceCodeTools.code.data.sourcetrail.sourcetrail_compute_function_diameter import compute_diameter
# from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import get_docstring, \
#     remove_offsets
from SourceCodeTools.nlp import create_tokenizer


import tokenize
from io import StringIO
def remove_comments_and_docstrings(source):
    """
    Returns 'source' minus comments and docstrings.
    https://stackoverflow.com/questions/1769332/script-to-remove-python-comments-docstrings
    """
    io_obj = StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        # The following two conditionals preserve indentation.
        # This is necessary because we're not using tokenize.untokenize()
        # (because it spits out code with copious amounts of oddly-placed
        # whitespace).
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        # Remove comments:
        if token_type == tokenize.COMMENT:
            pass
        # This series of conditionals removes docstrings:
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
        # This is likely a docstring; double-check we're not inside an operator:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        # Unlabelled indentation means we're inside an operator
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    return out


def process_code(source_file_content, node_resolver, mention_tokenizer):
    try:
        ast_processor = AstProcessor(source_file_content)
    except:
        logging.warning("Unknown exception")
        return None
    try: # TODO recursion error does not appear consistently. The issue is probably with library versions...
        edges = ast_processor.get_edges(as_dataframe=False)
    except RecursionError:
        return None

    if len(edges) == 0:
        return None

    edges = standardize_new_edges(edges, node_resolver, mention_tokenizer)

    return edges


def compute_gnn_passings(body, mention_tokenizer, num_layers=None):

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

    def compute_for_n_layers(n_layers):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)

        for node in G.nodes():
            non_zero_blocks = 0
            num_edges = 0
            loader = dgl.dataloading.NodeDataLoader(
                G, [node], sampler, batch_size=1, shuffle=True, num_workers=0)
            for input_nodes, seeds, blocks in loader:
                num_edges = 0
                for block in blocks:
                    if block.num_edges() > 0:
                        non_zero_blocks += 1
                        num_edges += block.num_edges()
            if num_edges != 0 and non_zero_blocks >= n_layers:
                break
        return num_edges

    passings_num_layers = compute_for_n_layers(num_layers)
    passings_diameter = compute_for_n_layers(diameter)

    return len(edges), passings_num_layers, passings_diameter


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
    mention_tokenizer = MentionTokenizer(args.bpe_path, create_subword_instances=False, connect_subwords=False)

    lengths_tr = defaultdict(list)
    lengths_gnn_layers = defaultdict(list)
    lengths_gnn_diameter = defaultdict(list)
    ratio_layers = []
    ratio_diameter = []

    for body in tqdm(bodies["body"]):
        if not has_valid_syntax(body):
            continue

        body_ = body
        body = body_.lstrip()
        initial_strip = body[:len(body_) - len(body)]

        # docsting_offsets = get_docstring(body)
        # body, replacements, docstrings = remove_offsets(body, [], docsting_offsets)
        body = remove_comments_and_docstrings(body)

        n_tokens = compute_transformer_passings(body, bpe)
        result = compute_gnn_passings(body, mention_tokenizer, args.num_layers)
        if result is None:
            continue
        n_edges, n_passings, n_passings_diam = result

        lengths_tr[n_tokens].append(n_tokens ** 2 * args.num_layers)
        lengths_gnn_layers[n_tokens].append(n_passings)# * args.num_layers)
        lengths_gnn_diameter[n_tokens].append(n_passings_diam)  # * args.num_layers)
        ratio_layers.append((n_tokens, n_passings))
        ratio_diameter.append((n_tokens, n_passings_diam))

    for key in lengths_tr:
        data_tr = np.array(lengths_tr[key])
        data_gnn_layers = np.array(lengths_gnn_layers[key])
        data_gnn_diameter = np.array(lengths_gnn_diameter[key])

        lengths_tr[key] = np.mean(data_tr)#, np.std(data_tr))
        lengths_gnn_layers[key] = np.mean(data_gnn_layers)#, np.std(data_gnn))
        lengths_gnn_diameter[key] = np.mean(data_gnn_diameter)

    data_ratios_layers = np.array(ratio_layers)
    data_ratios_diameter = np.array(ratio_diameter)

    plt.plot(data_ratios_layers[:, 0], data_ratios_layers[:, 1], "*")
    plt.plot(data_ratios_diameter[:, 0], data_ratios_diameter[:, 1], "*")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Message Exchanges")
    plt.legend([f"Layers = {args.num_layers}", "Layers = Graph Diameter"])
    plt.savefig(join(dirname(args.bodies), "tokens_edges.png"))
    plt.close()

    plt.hist(data_ratios_layers[:, 1] / data_ratios_layers[:, 0], bins=20)
    plt.hist(data_ratios_diameter[:, 1] / data_ratios_diameter[:, 0], bins=20)
    plt.xlabel("Number of edges / Number of tokens")
    plt.legend([f"Layers = {args.num_layers}", "Layers = Graph Diameter"])
    plt.savefig(join(dirname(args.bodies), "ratio.png"))
    plt.close()

    ratio_layers = data_ratios_layers[:, 1] / data_ratios_layers[:, 0]
    ratio_layers = (np.mean(ratio_layers), np.std(ratio_layers))

    ratio_diameter = data_ratios_diameter[:, 1] / data_ratios_diameter[:, 0]
    ratio_diameter = (np.mean(ratio_diameter), np.std(ratio_diameter))

    plt.plot(list(lengths_tr.keys()), np.array(list(lengths_tr.values())), "*")
    plt.plot(list(lengths_gnn_layers.keys()), np.array(list(lengths_gnn_layers.values())), "*")
    plt.plot(list(lengths_gnn_diameter.keys()), np.array(list(lengths_gnn_diameter.values())), "*")
    plt.gca().set_yscale('log')
    plt.grid()
    plt.legend([
        f"Transformer Layers = {args.num_layers}",
        f"GNN Layers = {args.num_layers}",
        f"GNN Layers = Graph Diameter"]
    )
    plt.xlabel("Number of Tokens")
    plt.ylabel("Number of Message Exchanges")
    plt.savefig(join(dirname(args.bodies), "avg_passings.png"))
    plt.close()





if __name__ == "__main__":
    main()