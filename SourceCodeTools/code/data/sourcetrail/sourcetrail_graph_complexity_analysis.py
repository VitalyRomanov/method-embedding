import logging
import pickle
from collections import defaultdict
from functools import partial
from os.path import join, dirname

import dgl
import pandas as pd
import argparse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

    unique_nodes = set(edges["source_node_id"]) | set(edges["target_node_id"])

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

    return len(edges), passings_num_layers, passings_diameter, diameter, len(unique_nodes)


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

    # lengths_tr = defaultdict(list)
    # lengths_gnn_layers = defaultdict(list)
    # lengths_gnn_diameter = defaultdict(list)
    # ratio_trans = []
    # ratio_layers = []
    # ratio_diameter = []
    # diameters = []
    # node_ratio = []
    #
    # for body in tqdm(bodies["body"]):
    #     if not has_valid_syntax(body):
    #         continue
    #
    #     body_ = body
    #     body = body_.lstrip()
    #     initial_strip = body[:len(body_) - len(body)]
    #
    #     # docsting_offsets = get_docstring(body)
    #     # body, replacements, docstrings = remove_offsets(body, [], docsting_offsets)
    #     body = remove_comments_and_docstrings(body)
    #
    #     n_tokens = compute_transformer_passings(body, bpe)
    #     result = compute_gnn_passings(body, mention_tokenizer, args.num_layers)
    #     if result is None:
    #         continue
    #     n_edges, n_passings, n_passings_diam, diameter, num_nodes = result
    #
    #     if num_nodes / n_tokens < 0.3:
    #         print()
    #
    #     # lengths_tr[n_tokens].append(n_tokens ** 2 * args.num_layers)
    #     # lengths_gnn_layers[n_tokens].append(n_passings)# * args.num_layers)
    #     # lengths_gnn_diameter[n_tokens].append(n_passings_diam)  # * args.num_layers)
    #     ratio_trans.append((n_tokens, n_tokens ** 2 * args.num_layers))
    #     ratio_layers.append((n_tokens, n_passings))
    #     ratio_diameter.append((n_tokens, n_passings_diam))
    #     node_ratio.append((n_tokens, num_nodes))
    #     diameters.append((n_tokens, diameter))
    #
    # for key in lengths_tr:
    #     data_tr = np.array(lengths_tr[key])
    #     data_gnn_layers = np.array(lengths_gnn_layers[key])
    #     data_gnn_diameter = np.array(lengths_gnn_diameter[key])
    #
    # # for key in lengths_tr:
    # #     data_tr = np.array(lengths_tr[key])
    # #     data_gnn_layers = np.array(lengths_gnn_layers[key])
    # #     data_gnn_diameter = np.array(lengths_gnn_diameter[key])
    # #
    # #     lengths_tr[key] = np.mean(data_tr)#, np.std(data_tr))
    # #     lengths_gnn_layers[key] = np.mean(data_gnn_layers)#, np.std(data_gnn))
    # #     lengths_gnn_diameter[key] = np.mean(data_gnn_diameter)
    #
    # data_node_ratio = np.array(node_ratio)
    # data_ratios_trans = np.array(ratio_trans)
    # data_ratios_layers = np.array(ratio_layers)
    # data_ratios_diameter = np.array(ratio_diameter)
    # data_diameters = np.array(diameters)

    get_path = partial(join, dirname(args.bodies))

    # np.save(get_path("data_node_ratio.npy"), data_node_ratio)
    # np.save(get_path("data_ratios_layers.npy"), data_ratios_layers)
    # np.save(get_path("data_ratios_diameter.npy"), data_ratios_diameter)
    # np.save(get_path("data_ratios_trans.npy"), data_ratios_trans)
    # np.save(get_path("data_diameters.npy"), data_diameters)
    # # pickle.dump(lengths_tr, open(get_path("lengths_tr.pkl"), "wb"))
    # # pickle.dump(lengths_gnn_layers, open(get_path("lengths_gnn_layers.pkl"), "wb"))
    # # pickle.dump(lengths_gnn_diameter, open(get_path("lengths_gnn_diameter.pkl"), "wb"))

    data_ratios_trans = np.load(get_path("data_ratios_trans.npy"))
    data_node_ratio = np.load(get_path("data_node_ratio.npy"))
    data_ratios_layers = np.load(get_path("data_ratios_layers.npy"))
    data_ratios_diameter = np.load(get_path("data_ratios_diameter.npy"))
    data_diameters = np.load(get_path("data_diameters.npy"))
    # lengths_tr = pickle.load(open(get_path("lengths_tr.pkl"), "rb"))
    # lengths_gnn_layers = pickle.load(open(get_path("lengths_gnn_layers.pkl"), "rb"))
    # lengths_gnn_diameter = pickle.load(open(get_path("lengths_gnn_diameter.pkl"), "rb"))

    # plt.plot(data_ratios_layers[:, 0], data_ratios_layers[:, 1], "*")
    # plt.plot(data_ratios_diameter[:, 0], data_ratios_diameter[:, 1], "*")
    # plt.xlabel("Число токенов")
    # plt.ylabel("Число обменов сообщениями")
    # plt.legend([f"Количество слоёв = {args.num_layers}", "Количество слоёв = Диаметр графа"])
    # plt.savefig(join(dirname(args.bodies), "tokens_edges.pdf"))
    # plt.close()

    print("Average diameter:", sum(data_diameters[:, 1]) / len(data_diameters[:, 1]))

    plt.hexbin(data_diameters[:, 0], data_diameters[:, 1], bins="log", cmap="magma_r", gridsize=150)
    plt.xlabel("Число токенов в исходном коде")
    plt.ylabel("Диметр графа исходного кода")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend([f"Количество слоёв = {args.num_layers}", "Количество слоёв = Диаметр графа"])
    plt.savefig(join(dirname(args.bodies), "token_diameters.pdf"))
    plt.close()

    plt.hist(data_node_ratio[:, 1] / data_node_ratio[:, 0], bins=20, range=(0, 2), alpha=0.7)
    plt.xlabel("Число узлов в графе / Число токенов в исходном коде")
    plt.ylabel("Частота")
    # plt.legend([f"Количество слоёв = {args.num_layers}", "Количество слоёв = Диаметр графа"])
    plt.savefig(join(dirname(args.bodies), "ratio_nodes.pdf"))
    plt.close()

    # plt.hist(data_ratios_layers[:, 1] / data_ratios_layers[:, 0], bins=20, range=(0,30), alpha=0.7)
    # plt.hist(data_ratios_diameter[:, 1] / data_ratios_diameter[:, 0], bins=20, range=(0,30), alpha=0.7)
    # plt.xlabel("Число обменов сообщениями / Число токенов")
    # plt.ylabel("Частота")
    # plt.legend([f"Количество слоёв = {args.num_layers}", "Количество слоёв = Диаметр графа"])
    # plt.savefig(join(dirname(args.bodies), "ratio.pdf"))
    # plt.close()

    # ratio_layers = data_ratios_layers[:, 1] / data_ratios_layers[:, 0]
    # ratio_layers = (np.mean(ratio_layers), np.std(ratio_layers))
    #
    # ratio_diameter = data_ratios_diameter[:, 1] / data_ratios_diameter[:, 0]
    # ratio_diameter = (np.mean(ratio_diameter), np.std(ratio_diameter))

    plt.figure()
    plt.xlim([0, 1750])
    plt.ylim([0, 5e4])
    plt.plot(data_ratios_trans[:, 0], data_ratios_trans[:, 1], "o", alpha=0.3, markersize=2, markeredgewidth=0)
    plt.plot(data_ratios_layers[:, 0], data_ratios_layers[:, 1], "o",  alpha=0.3, markersize=2, markeredgewidth=0)
    plt.plot(data_ratios_diameter[:, 0], data_ratios_diameter[:, 1], "o",  alpha=0.3, markersize=2, markeredgewidth=0)
    # plt.gca().set_yscale('log')
    plt.grid()
    plt.legend([
        f"Трансформер, количество слоёв,  = {args.num_layers}",
        f"GNN, количество слоёв = {args.num_layers}",
        f"GNN, количество слоёв = Диаметр графа"]
    )
    plt.xlabel("Число токенов")
    plt.ylabel("Число обменов сообщениями")
    plt.savefig(join(dirname(args.bodies), "avg_passings.pdf"))
    plt.close()

    plt.plot([], [], ".")
    plt.plot(data_node_ratio[:, 1], data_ratios_layers[:, 1], ".", alpha=0.1)
    plt.plot(data_node_ratio[:, 1], data_ratios_diameter[:, 1], ".", alpha=0.1)
    # plt.gca().set_yscale('log')
    plt.ylim([0, 1e4])
    plt.grid()
    plt.legend([
        "",
        f"Количество слоёв, GNN = {args.num_layers}",
        f"Количество слоёв, GNN = Диаметр графа"]
    )
    plt.xlabel("Число узлов в графе исходного кода")
    plt.ylabel("Число обменов сообщениями")
    plt.savefig(join(dirname(args.bodies), "avg_passings_vs_nodes.pdf"))
    plt.close()





if __name__ == "__main__":
    main()