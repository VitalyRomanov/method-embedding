import ast
import re
from copy import copy

from nltk import RegexpTokenizer

from SourceCodeTools.code.ast.python_ast import AstGraphGenerator, GNode, PythonSharedNodes
from SourceCodeTools.code.annotator_utils import to_offsets, overlap, resolve_self_collision
from SourceCodeTools.code.data.file_utils import *
from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
from SourceCodeTools.code.data.sourcetrail.common import custom_tqdm

pd.options.mode.chained_assignment = None


class NodeResolver:
    def __init__(self, nodes):

        self.nodeid2name = dict(zip(nodes['id'].tolist(), nodes['serialized_name'].tolist()))
        self.nodeid2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        self.valid_new_node = nodes['id'].max() + 1
        self.node_ids = {}
        self.new_nodes = []

        self.old_nodes = nodes.copy()
        self.old_nodes['mentioned_in'] = pd.NA
        self.old_nodes = self.old_nodes.astype({'mentioned_in': 'Int32'})

    def get_new_node_id(self):
        new_id = self.valid_new_node
        self.valid_new_node += 1
        return new_id

    def resolve(self, node, srctrl2original):

        decorated = "@" in node.name
        assert len([c for c in node.name if c == "@"]) <= 1

        if decorated:
            name_, decorator = node.name.split("@")
        else:
            name_ = copy(node.name)

        if name_ in srctrl2original and node.type != "type_annotation":
            node_id = int(name_.split("_")[1])
            # the only types that should appear here are "Name", "mention", "#attr#"
            # the first two are from mentions, and the last one is when references sourcetrail node is an attribute
            # if node.type not in {"Name", "mention", "#attr#"}:
            if node.type in {"#keyword#"}:
                # TODO
                # either a sourcetrail error or the error parsing
                return GNode(name=srctrl2original[name_], type=node.type)
            assert node.type in {"Name", "mention", "#attr#"}
            real_name = srctrl2original[name_]
            global_name = self.nodeid2name[node_id]
            global_type = self.nodeid2type[node_id]

            if node.type == "Name":
                # name always go together with mention, therefore no global reference in Name
                new_node = GNode(name=real_name, type=node.type, global_id=node_id)
                # new_node = node
            else:
                if decorated:
                    assert node.type == "mention"
                    real_name += "@" + decorator
                    type_ = "mention"
                else:
                    assert node.type == "#attr#"
                    type_ = node.type
                new_node = GNode(name=real_name, type=type_, global_name=global_name, global_id=node_id, global_type=global_type)

            # if decorated:
            #     # replace the sourcetrail node with its local name that have originally appeared in the
            #     # source code of the function, because this was a mention, and should stay a mention
            #     real_name = srctrl2original[name_]
            #     real_name += "@" + decorator
            #     type_ = "mention"
            #     new_node = GNode(name=real_name, type=type_, name_scope="local")
            # else:
            #     # replace the sourcetrail node with full path
            #     real_name = self.nodeid2name[node_id]  # full name, including modules
            #     type_ = self.nodeid2type[node_id]
            #     new_node = GNode(name=real_name, type=type_, id=node_id, name_scope="global")
            assert "srctrlnd" not in new_node.name
            return new_node

        replacements = dict()
        for name in re.finditer("srctrlnd_[0-9]+", name_):
            if isinstance(name, re.Match):
                name = name.group()
            elif isinstance(name, str):
                pass
            else:
                print("Unknown type")
            if name.startswith("srctrlnd_"):
                node_id = name.split("_")[1]
                replacements[name] = {
                    "name": self.nodeid2name[int(node_id)],
                    "id": node_id
                }

        real_name = name_
        for r, v in replacements.items():
            real_name = real_name.replace(r, v["name"])

        if decorated:
            real_name += "@" + decorator

        assert "srctrlnd" not in real_name

        return GNode(name=real_name, type=node.type)

    def resolve_node_id(self, node, function_id):
        if not hasattr(node, "id"):
            node_repr = (node.name.strip(), node.type.strip())

            if node_repr in self.node_ids:
                node_id = self.node_ids[node_repr]
                node.setprop("id", node_id)
            else:
                new_id = self.get_new_node_id()
                self.node_ids[node_repr] = new_id

                if not PythonSharedNodes.is_shared(node):
                    assert "0x" in node.name

                self.new_nodes.append(
                    {
                        "id": new_id,
                        "type": node.type,
                        "serialized_name": node.name,
                        "mentioned_in": function_id if not PythonSharedNodes.is_shared(node) else pd.NA
                    }
                )

                # temp = pd.DataFrame(self.new_nodes)
                # temp['node_repr'] = list(zip(temp['serialized_name'], temp['type']))
                # assert len(temp) == len(set(temp['node_repr'].to_list()))

                node.setprop("id", new_id)
        return node

    def prepare_for_write(self):
        nodes = pd.concat([self.old_nodes, self.new_nodes_for_write()])[
            ['id', 'type', 'serialized_name', 'mentioned_in']
        ]

        return nodes

    def new_nodes_for_write(self):

        new_nodes = pd.DataFrame(self.new_nodes)[
            ['id', 'type', 'serialized_name', 'mentioned_in']
        ]

        return new_nodes

    def get_mention_edges(self):
        mention_edges = None

        mentined_nodes = pd.DataFrame(self.new_nodes).query("not mentioned_in.isnull()")
        mention_groups = mentined_nodes.groupby("mentioned_in")

        for grp_val, group in custom_tqdm(mention_groups, total=len(mention_groups), message="Preparing mention edges"):
            mention_edges_direct = group[['id']]
            mention_edges_direct['dst'] = grp_val
            mention_edges_direct['type'] = "mention_scope"
            mention_edges_direct = mention_edges_direct.rename({"id": "src"}, axis=1)

            mention_edges_rev = mention_edges_direct.copy()
            mention_edges_rev = mention_edges_rev.rename({"src": "dst", "dst": "src"}, axis=1)
            mention_edges_rev['type'] = "mention_scope_rev"

            all_mention_edges = mention_edges_direct.append(mention_edges_rev)
            all_mention_edges['mentioned_in'] = grp_val

            if mention_edges is None:
                mention_edges = all_mention_edges
            else:
                mention_edges = mention_edges.append(all_mention_edges)

        return mention_edges

        # for node in self.new_nodes:
        #     if "mentioned_in" in node:
        #         mentioned_in = node['mentioned_in']
        #         if isinstance(mentioned_in, int):
        #             mention_edges.append({
        #                 "src": node["id"],
        #                 "dst": mentioned_in,
        #                 "type": "mention_scope",
        #                 'line': pd.NA,
        #                 'end_line': pd.NA,
        #                 'col_offset': pd.NA,
        #                 'end_col_offset': pd.NA,
        #             })
        #             mention_edges.append(make_reverse_edge(mention_edges[-1]))
        #
        # return mention_edges


def get_ast_nodes(edges):
    nodes = []
    for ind, row in edges.iterrows():
        if "line" not in row or pd.isna(row["line"]):
            continue

        nodes.append((
            row['line'],
            row['end_line'],
            row['col_offset'],
            row['end_col_offset'],
            row['src']
        ))

    return nodes


from SourceCodeTools.code.annotator_utils import adjust_offsets


def format_replacement_offsets(offsets):
    return [(offset[0], offset[0], offset[1], offset[2], offset[3]) for offset in offsets]


def keep_node(node_string):
    if "(" in node_string or ")" in node_string or "[" in node_string or "]" in node_string or \
            "{" in node_string or "}" in node_string or " " in node_string or "," in node_string:
        return False
    return True


def filter_nodes(offsets, body):
    """
    Prevents overlaps between nodes extracted from AST and nodes provided by Sourcetrail
    """
    return [offset for offset in offsets if keep_node(body[offset[0]:offset[1]])]


def join_offsets(offsets_1, offsets_2):
    joined = []
    while offsets_1 or offsets_2:
        if len(offsets_1) == 0:
            joined.append(offsets_2.pop(0))
        elif len(offsets_2) == 0:
            joined.append(offsets_1.pop(0))
        elif offsets_1[0] == offsets_2[0]:
            joined.append(offsets_1.pop(0))
            offsets_2.pop(0)
        elif overlap(offsets_1[0], offsets_2[0]):
            # Exception: ('Should not overlap:', (360, 382, 611771), (368, 375, 611758))
            # >>> body[360:382]
            # Out[1]: 'ZIIwSfl(JicFbMT, item)'
            # >>> body[368:375]
            # Out[3]: 'JicFbMT'
            #
            # it appears some nodes can overlap. preserve the smallest one
            len_1 = offsets_1[0][1] - offsets_1[0][0]
            len_2 = offsets_2[0][1] - offsets_2[0][0]
            if len_1 < len_2:
                joined.append(offsets_1.pop(0))
                offsets_2.pop(0)
            elif len_2 > len_1:
                joined.append(offsets_2.pop(0))
                offsets_1.pop(0)
            else:
                # print("Should not overlap:", offsets_1[0], offsets_2[0])
                # TODO
                # it seems to be some unreasonable error. skip both versions
                # joined.append(offsets_1.pop(0))
                offsets_1.pop(0)
                offsets_2.pop(0)
                # raise Exception("Should not overlap:", offsets_1[0], offsets_2[0])
        elif offsets_1[0][0] < offsets_2[0][0]:
            joined.append(offsets_1.pop(0))
        elif offsets_1[0][0] > offsets_2[0][0]:
            joined.append(offsets_2.pop(0))
        else:
            raise Exception("Illegal scenario")

    return joined


def random_replacement_lookup(name, type, replacements, tokenizer):
    # TODO
    # next line is a work around before random replacements are phased out
    if type not in {"Name", "mention", "#attr#"}: return name
    if "[" not in name: # Name, mention
        return replacements.get(name, name)
    else:
        tokens = tokenizer.tokenize(name)
        r_tokens = map(lambda x: replacements.get(x, x), tokens)
        corrected_name = "".join(r_tokens)
        return corrected_name


# def filter_out_mentions_for_srctrl_nodes(edges):
#     # filter mention_scope edges for sourcetrail nodes
#
#     def srctrl_decode(x):
#         return GNode(name=x.name.split("@")[0], type="Name") if x.name.startswith("srctrlnd") and "@" in x.name else x
#
#     edges['src'] = edges['src'].apply(srctrl_decode)
#     edges['dst'] = edges['dst'].apply(srctrl_decode)
#     edges = edges.query("src!=dst")
#     edges['help'] = edges['dst'].apply(lambda x: x.name.split("_")[0])
#     # edges = edges.query("help!='srctrlnd' or type!='mention_scope'").drop("help", axis=1)
#     edges = edges.query("help!='srctrlnd'").drop("help", axis=1)
#     return edges


def produce_subword_edges(subwords, dst, connect_subwords=False):
    new_edges = []

    subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
    for ind, subword in enumerate(subwords):
        new_edges.append({
            'src': subword,
            'dst': dst,
            'type': 'subword',
            'line': pd.NA,
            'end_line': pd.NA,
            'col_offset': pd.NA,
            'end_col_offset': pd.NA,
        })
        if connect_subwords:
            if ind < len(subwords) - 1:
                new_edges.append({
                    'src': subword,
                    'dst': subwords[ind + 1],
                    'type': 'next_subword',
                    'line': pd.NA,
                    'end_line': pd.NA,
                    'col_offset': pd.NA,
                    'end_col_offset': pd.NA,
                })
            if ind > 0:
                new_edges.append({
                    'src': subword,
                    'dst': subwords[ind - 1],
                    'type': 'prev_subword',
                    'line': pd.NA,
                    'end_line': pd.NA,
                    'col_offset': pd.NA,
                    'end_col_offset': pd.NA,
                })

    return new_edges


def produce_subword_edges_with_instances(subwords, dst):
    new_edges = []

    subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
    instances = list(map(lambda x: GNode(name=x.name + "@" + dst.name, type="subword_instance"), subwords))
    for ind, subword in enumerate(subwords):
        subword_instance = instances[ind]
        new_edges.append({
            'src': subword,
            'dst': subword_instance,
            'type': 'subword_instance',
            'line': pd.NA,
            'end_line': pd.NA,
            'col_offset': pd.NA,
            'end_col_offset': pd.NA,
        })
        new_edges.append({
            'src': subword_instance,
            'dst': dst,
            'type': 'subword',
            'line': pd.NA,
            'end_line': pd.NA,
            'col_offset': pd.NA,
            'end_col_offset': pd.NA,
        })
        if ind < len(subwords) - 1:
            new_edges.append({
                'src': subword_instance,
                'dst': instances[ind + 1],
                'type': 'next_subword',
                'line': pd.NA,
                'end_line': pd.NA,
                'col_offset': pd.NA,
                'end_col_offset': pd.NA,
            })
        if ind > 0:
            new_edges.append({
                'src': subword_instance,
                'dst': instances[ind - 1],
                'type': 'prev_subword',
                'line': pd.NA,
                'end_line': pd.NA,
                'col_offset': pd.NA,
                'end_col_offset': pd.NA,
            })

    return new_edges


def global_mention_edges(edge):
    edge['type'] = "global_mention"
    return [edge, make_reverse_edge(edge)]


def global_mention_edges_from_node(node):
    global_mention = {
        "src": GNode(name=node.global_name, type=node.global_type, id=node.global_id),
        "dst": node,
        "type": "global_mention",
        'line': pd.NA,
        'end_line': pd.NA,
        'col_offset': pd.NA,
        'end_col_offset': pd.NA,
    }
    return [global_mention, make_reverse_edge(global_mention)]


def make_reverse_edge(edge):
    rev_edge = copy(edge)
    rev_edge['type'] = edge['type'] + "_rev"
    rev_edge['src'] = edge['dst']
    rev_edge['dst'] = edge['src']
    rev_edge['offsets'] = None
    return rev_edge


def replace_mentions_with_subword_instances(edges, bpe, create_subword_instances, connect_subwords):
    edges = edges.to_dict(orient="records")

    if create_subword_instances:
        def produce_subw_edges(subwords, dst):
            return produce_subword_edges_with_instances(subwords, dst)
    else:
        def produce_subw_edges(subwords, dst):
            return produce_subword_edges(subwords, dst, connect_subwords)

    new_edges = []
    for edge in edges:
        if edge['src'].type == "#attr#":
            if hasattr(edge['src'], "global_name"):
                new_edges.extend(global_mention_edges_from_node(edge['src']))
        elif edge['dst'].type == "mention":
            if hasattr(edge['dst'], "global_name"):
                new_edges.extend(global_mention_edges_from_node(edge['dst']))

        if edge['type'] == "local_mention":
            # is_global_mention = hasattr(edge['src'], "id")
            # if is_global_mention:
            #     # this edge connects sourcetrail node need to add couple of links
            #     # to ensure global information flow
            #     new_edges.extend(global_mention_edges(edge))

            dst = edge['dst']

            if bpe is not None:
                if hasattr(dst, "name_scope") and dst.name_scope == "local":
                    subwords = bpe(dst.name.split("@")[0])
                else:
                    subwords = bpe(edge['src'].name)

                new_edges.extend(produce_subw_edges(subwords, dst))
            else:
                new_edges.append(edge)

        elif bpe is not None and (
                (
                    edge['src'].type in PythonSharedNodes.tokenizable_types
                ) or (
                    edge['dst'].type in {"Global"} and edge['src'].type != "Constant"
                )):
            new_edges.append(edge)
            new_edges.append(make_reverse_edge(edge))

            dst = edge['src']
            subwords = bpe(dst.name)
            new_edges.extend(produce_subw_edges(subwords, dst))
        else:
            new_edges.append(edge)

    return pd.DataFrame(new_edges)


def get_srctrl2original_replacements(record):
    random2srctrl = record['random_2_srctrl']
    random2original = record['random_2_original']

    return {random2srctrl[key]: random2original[key] for key in random2original}


def append_edges(ast_edges, new_edges):
    if ast_edges is None:
        return new_edges[['id', 'type', 'src', 'dst', 'mentioned_in']]
    else:
        return ast_edges.append(new_edges[['id', 'type', 'src', 'dst', 'mentioned_in']])


def write_bodies(path, bodies):
    write_pickle(pd.DataFrame(bodies), path)


def write_nodes(path, node_resolver: NodeResolver):
    # new_nodes = pd.concat([node_resolver.old_nodes, pd.DataFrame(node_resolver.new_nodes)])[
    #     ['id', 'type', 'serialized_name', 'mentioned_in']
    # ]

    new_nodes = node_resolver.prepare_for_write()
    write_pickle(new_nodes, path)


def leaf_nodes_are_leaf_types(nodes: pd.DataFrame, edges: pd.DataFrame):
    leaf = set(edges["source_node_id"].tolist()) - set(edges["target_node_id"].tolist())
    leaf_nodes = nodes[nodes["id"].apply(lambda id_: id_ in leaf)]

    if len(leaf_nodes.query("type == 'subword'")) > 0:
        leaf_types = PythonSharedNodes.subword_leaf_types
    else:
        leaf_types = PythonSharedNodes.named_leaf_types

    _leaf_nodes = leaf_nodes[
        leaf_nodes['type'].apply(lambda type_: type_ not in leaf_types)
    ]

    if len(_leaf_nodes) > 0:
        logging.warning(f"Not a leaf type: not in {leaf_types}")
        logging.warning(_leaf_nodes.to_string())

    # is_leaf = map(lambda type_: type_ in PythonSharedNodes.leaf_types, leaf_nodes['type'].unique())
    # return all(is_leaf)


def _get_from_ast(bodies, node_resolver, bpe_tokenizer_path=None, create_subword_instances=True, connect_subwords=False):
    ast_edges = None

    bodies_with_replacements = {}

    subword_tokenizer = make_tokenizer(load_bpe_model((bpe_tokenizer_path))) \
        if bpe_tokenizer_path else None

    tokenizer = RegexpTokenizer("\w+|[^\w\s]")

    for ind_bodies, (_, row) in custom_tqdm(
            enumerate(bodies.iterrows()), message="Extracting AST edges", total=len(bodies)
    ):
        orig_body = row['body_with_random_replacements']
        if not isinstance(orig_body, str):
            continue

        srctrl2original = get_srctrl2original_replacements(row)

        c = orig_body.lstrip()
        strip_len = len(orig_body) - len(c)

        try:
            ast.parse(c)
        except SyntaxError as e:
            print(e)
            continue

        replacements = row['random_2_srctrl']

        g = AstGraphGenerator(c)

        edges = g.get_edges()

        if len(edges) == 0:
            continue

        # replacements_lookup = lambda x: complex_replacement_lookup(x, replacements)
        replacements_lookup = lambda x: \
            GNode(name=random_replacement_lookup(x.name, x.type, replacements, tokenizer),
                  type=x.type) if "@" not in x.name else \
                GNode(name=random_replacement_lookup(x.name.split("@")[0], x.type, replacements, tokenizer) +
                           "@" + x.name.split("@")[1],
                      type=x.type)

        edges['src'] = edges['src'].apply(replacements_lookup)
        edges['dst'] = edges['dst'].apply(replacements_lookup)

        resolve = lambda node: node_resolver.resolve(node, srctrl2original)

        edges['src'] = edges['src'].apply(resolve)
        edges['dst'] = edges['dst'].apply(resolve)

        edges = replace_mentions_with_subword_instances(
            edges, subword_tokenizer, create_subword_instances=create_subword_instances,
            connect_subwords=connect_subwords
        )

        resolve_node_id = lambda node: node_resolver.resolve_node_id(node, row['id'])

        edges['src'] = edges['src'].apply(resolve_node_id)
        edges['dst'] = edges['dst'].apply(resolve_node_id)

        extract_id = lambda node: node.id
        edges['src'] = edges['src'].apply(extract_id)
        edges['dst'] = edges['dst'].apply(extract_id)

        # edges = edges.append(node_resolver.get_mention_edges())
        edges = edges.drop_duplicates(subset=["src", "dst", "type"])

        edges['id'] = 0

        ast_nodes = resolve_self_collision(filter_nodes(adjust_offsets(
            to_offsets(c, get_ast_nodes(edges), as_bytes=True), -strip_len), orig_body))

        srctrl_nodes = list(map(
            lambda x: (x[0], x[1], node_resolver.resolve(GNode(name=x[2], type="Name"), srctrl2original).global_id),
            to_offsets(row['body_with_random_replacements'],
                       format_replacement_offsets(row['replacement_list']))
        ))

        all_offsets = join_offsets(
            sorted(ast_nodes, key=lambda x: x[0]),
            sorted(srctrl_nodes, key=lambda x: x[0])
        )

        bodies_with_replacements[row['id']] = all_offsets

        # append_edges(path=edges_with_ast_name, edges=edges)
        edges['mentioned_in'] = row['id']
        ast_edges = append_edges(ast_edges=ast_edges, new_edges=edges)
        # print("\r%d/%d" % (ind_bodies, len(bodies['body_normalized'])), end="")

    # print(" " * 30, end="\r")

    bodies['graph_node_replacements'] = bodies['id'].apply(lambda id_: bodies_with_replacements.get(id_, None))

    # write_nodes(path=nodes_with_ast_name, node_resolver=node_resolver)

    # ast_nodes = pd.DataFrame(node_resolver.new_nodes)[['id', 'type', 'serialized_name', 'mentioned_in']].astype(
    #     {'mentioned_in': 'Int32'}
    # )

    ast_edges = ast_edges.append(node_resolver.get_mention_edges())
    ast_edges['id'] = 0

    ast_nodes = node_resolver.new_nodes_for_write()
    ast_edges = ast_edges.rename({'src': 'source_node_id', 'dst': 'target_node_id'}, axis=1).astype(
        {'mentioned_in': 'Int32'}
    )

    # assert leaf_nodes_are_leaf_types(ast_nodes, ast_edges)
    leaf_nodes_are_leaf_types(ast_nodes, ast_edges)

    return ast_nodes, ast_edges, bodies


def get_from_ast(nodes, bodies, bpe_tokenizer_path, create_subword_instances, connect_subwords):

    node_resolver = NodeResolver(nodes)

    return _get_from_ast(bodies, node_resolver,
                bpe_tokenizer_path=bpe_tokenizer_path,
                create_subword_instances=create_subword_instances,
                connect_subwords=connect_subwords)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert python funcitons to graphs. Include subwords for names.')
    parser.add_argument('working_directory', type=str,
                        help='Path to ')
    parser.add_argument('--bpe_tokenizer', '-bpe', dest='bpe_tokenizer', type=str,
                        help='')
    parser.add_argument('--create_subword_instances', action='store_true', default=False, help="")
    parser.add_argument('--connect_subwords', action='store_true', default=False,
                        help="Takes effect only when `create_subword_instances` is False")

    args = parser.parse_args()

    working_directory = args.working_directory
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)
    bodies = read_processed_bodies(working_directory)

    ast_nodes, ast_edges, bodies = get_from_ast(nodes, bodies, args.bpe_tokenizer,
                                                args.create_subword_instances, args.connect_subwords)

    edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.bz2")
    nodes_with_ast_name = os.path.join(working_directory, "nodes_with_ast.bz2")

    persist(nodes.append(ast_nodes), nodes_with_ast_name)
    persist(edges.append(ast_edges), edges_with_ast_name)
    write_processed_bodies(bodies, working_directory)


