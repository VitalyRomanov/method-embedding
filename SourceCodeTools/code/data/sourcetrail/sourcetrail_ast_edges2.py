import argparse
import re
from copy import copy
from time import time_ns

import networkx as nx

from SourceCodeTools.code.data.sourcetrail.common import *
from SourceCodeTools.code.data.sourcetrail.sourcetrail_add_reverse_edges import add_reverse_edges
from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges import NodeResolver, make_reverse_edge
from SourceCodeTools.code.python_ast import AstGraphGenerator, GNode, PythonSharedNodes
from SourceCodeTools.nlp.entity.annotator.annotator_utils import adjust_offsets2
from SourceCodeTools.nlp.entity.annotator.annotator_utils import overlap as range_overlap
from SourceCodeTools.nlp.entity.annotator.annotator_utils import to_offsets, get_cum_lens
from SourceCodeTools.nlp.string_tools import get_byte_to_char_map
from SourceCodeTools.code.data.sourcetrail.sourcetrail_parse_bodies2 import has_valid_syntax


class MentionTokenizer:
    def __init__(self, bpe_tokenizer_path, create_subword_instances, connect_subwords):
        from SourceCodeTools.nlp.embed.bpe import make_tokenizer
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model

        self.bpe = make_tokenizer(load_bpe_model(bpe_tokenizer_path)) \
            if bpe_tokenizer_path else None
        self.create_subword_instances = create_subword_instances
        self.connect_subwords = connect_subwords

    def replace_mentions_with_subwords(self, edges):

        if self.create_subword_instances:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges_with_instances(subwords, dst)
        else:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges(subwords, dst, self.connect_subwords)

        new_edges = []
        for edge in edges:
            # if edge['src'].type in {"#attr#", "Name"}:
            #     if hasattr(edge['src'], "global_id"):
            #         new_edges.extend(self.global_mention_edges_from_node(edge['src']))
            # elif edge['dst'].type == "mention":
            #     if hasattr(edge['dst'], "global_id"):
            #         new_edges.extend(self.global_mention_edges_from_node(edge['dst']))

            if edge['type'] == "local_mention":

                dst = edge['dst']

                if self.bpe is not None:
                    if hasattr(dst, "name_scope") and dst.name_scope == "local":
                        # TODO
                        # this rule seems to be irrelevant now
                        subwords = self.bpe(dst.name.split("@")[0])
                    else:
                        subwords = self.bpe(edge['src'].name)

                    new_edges.extend(produce_subw_edges(subwords, dst))
                else:
                    new_edges.append(edge)

            elif self.bpe is not None and edge["type"] == "__global_name":
                subwords = self.bpe(edge['src'].name)
                new_edges.extend(produce_subw_edges(subwords, edge['dst']))
            elif self.bpe is not None and (
                    (
                            edge['src'].type in PythonSharedNodes.tokenizable_types_and_annotations  #tokenizable_types
                    ) or (
                    edge['dst'].type in {"Global"} and edge['src'].type != "Constant")
            ):
                new_edges.append(edge)
                new_edges.append(make_reverse_edge(edge))

                dst = edge['src']
                subwords = self.bpe(dst.name)
                new_edges.extend(produce_subw_edges(subwords, dst))
            else:
                new_edges.append(edge)

        return new_edges

    # @staticmethod
    # def global_mention_edges_from_node(node):
    #     global_edges = []
    #     if type(node.global_id) is int:
    #         id_type = [(node.global_id, node.global_type)]
    #     else:
    #         id_type = zip(node.global_id, node.global_type)
    #
    #     for gid, gtype in id_type:
    #         global_mention = {
    #             "src": GNode(name=None, type=gtype, id=gid),
    #             "dst": node,
    #             "type": "global_mention",
    #             "offsets": None
    #         }
    #         global_edges.append(global_mention)
    #         global_edges.append(make_reverse_edge(global_mention))
    #     return global_edges

    @staticmethod
    def connect_prev_next_subwords(edges, current, prev_subw, next_subw):
        if next_subw is not None:
            edges.append({
                'src': current,
                'dst': next_subw,
                'type': 'next_subword',
                'offsets': None
            })
        if prev_subw is not None:
            edges.append({
                'src': current,
                'dst': prev_subw,
                'type': 'prev_subword',
                'offsets': None
            })

    def produce_subword_edges(self, subwords, dst, connect_subwords=False):
        new_edges = []

        subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
        for ind, subword in enumerate(subwords):
            new_edges.append({
                'src': subword,
                'dst': dst,
                'type': 'subword',
                'offsets': None
            })
            if connect_subwords:
                self.connect_prev_next_subwords(new_edges, subword, subwords[ind - 1] if ind > 0 else None,
                                                subwords[ind + 1] if ind < len(subwords) - 1 else None)
        return new_edges

    def produce_subword_edges_with_instances(self, subwords, dst, connect_subwords=True):
        new_edges = []

        subwords = list(map(lambda x: GNode(name=x, type="subword"), subwords))
        instances = list(map(lambda x: GNode(name=x.name + "@" + dst.name, type="subword_instance"), subwords))
        for ind, subword in enumerate(subwords):
            subword_instance = instances[ind]
            new_edges.append({
                'src': subword,
                'dst': subword_instance,
                'type': 'subword_instance',
                'offsets': None
            })
            new_edges.append({
                'src': subword_instance,
                'dst': dst,
                'type': 'subword',
                'offsets': None
            })
            if connect_subwords:
                self.connect_prev_next_subwords(new_edges, subword, subwords[ind - 1] if ind > 0 else None,
                                                subwords[ind + 1] if ind < len(subwords) - 1 else None)
        return new_edges


class GlobalNodeMatcher:
    def __init__(self, nodes, edges):
        from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types
        # filter function, classes, methods, and modules
        self.allowed_node_types = set([node_types[tt] for tt in [8, 128, 4096, 8192]])
        self.allowed_edge_types = {"defines_rev"}

        self.allowed_ast_node_types = {"FunctionDef", "ClassDef", "Module", "mention"}
        self.allowed_ast_edge_types = {"function_name", "class_name", "global_mention_rev"}

        self.nodes = nodes[nodes['type'].apply(lambda type: type in self.allowed_node_types)]
        self.edges = edges[edges["type"].apply(lambda type: type in self.allowed_edge_types)]

        self.global_names = self.get_node_names(self.nodes)
        self.global_types = self.get_node_types(self.nodes)

        self.global_graph = nx.DiGraph()
        self.global_graph.add_edges_from(self.get_edges_for_graph(self.edges))

    @staticmethod
    def get_edges_for_graph(edges):
        try:
            edge_info = zip(edges['src'], edges['dst'], edges['type'])
        except KeyError:
            edge_info = zip(edges['source_node_id'], edges['target_node_id'], edges['type'])
        edge_info = map(lambda x: (x[0], x[1], {"type": x[2]}), edge_info)
        return edge_info

    @staticmethod
    def get_node_names(nodes):
        id2name = dict(zip(nodes["id"], nodes["serialized_name"]))
        return id2name

    @staticmethod
    def get_node_types(nodes):
        id2type = dict(zip(nodes["id"], nodes["type"]))
        return id2type

    def match_with_global_nodes(self, nodes, edges):
        nodes = pd.DataFrame([node for node in nodes if node["type"] in self.allowed_ast_node_types])
        edges = pd.DataFrame([edge for edge in edges if edge["type"] in self.allowed_ast_edge_types])

        if len(edges) == 0:
            return {}

        func_nodes = nodes.query("type == 'FunctionDef'")["id"].values
        class_nodes = nodes.query("type == 'ClassDef'")["id"].values
        module_nodes = nodes.query("type == 'Module'")["id"].values

        g = nx.DiGraph()
        g.add_edges_from(self.get_edges_for_graph(edges))

        def find_global_id(graph, def_id, motif):
            """
            Do function or class global id lookup
            :param graph:
            :param def_id:
            :param motif: list with edge types, return None if path does not exist
            :return:
            """
            name_node = def_id
            motif = copy(motif)
            while len(motif) > 0:
                link_type = motif.pop(0)
                for node, eprop in graph[name_node].items():
                    if eprop["type"] == link_type:
                        name_node = node
                        break
                else:
                    return None
            return name_node

        new_node_ids = {}
        module_candidates = []

        def get_global_id_and_module_candidates(nodes, paths):
            """
            Get global ids for function or class definitions and get the candidate module location
            where they were defined.
            :param nodes: list of nodes to inspect, should be of the same type
            :param paths: path of edge types that will lead to the global node
            :return: mapping from ast node id to global id, the list of candidate modules
            """
            new_node_ids = {}
            module_candidates = []
            for node in nodes:
                gid = find_global_id(g, node, paths)
                if gid is None:
                    continue
                new_node_ids[node] = gid
                # new_node_ids[gid] = node
                module_candidate = find_global_id(self.global_graph, gid, ["defines_rev"])
                if module_candidate is not None and module_candidate in self.global_types and self.global_types[module_candidate] == "module":
                    module_candidates.append(module_candidate)
            return new_node_ids, module_candidates

        def get_global_module_id(module_nodes, module_candidates, func_global, class_global):
            assert len(module_nodes) == 1
            from collections import Counter
            if len(module_candidates) > 0:
                mod_id, count = Counter(module_candidates).most_common(1)[0]
                return {module_nodes[0]: mod_id}
            mod_id = get_global_module_id2(func_global, class_global)
            if mod_id is None:
                return {}
            else:
                return {module_nodes[0]: mod_id}
                # new_node_ids[module_nodes[0]] = mod_id
                # new_node_ids[mod_id] = module_nodes[0]

        def get_global_module_id2(func_global, class_global):
            classes = class_global.values()
            candidate_names = list(map(lambda id_: ".".join(self.global_names[id_].split(".")[:-1]), classes))

            functions = func_global.values()
            candidate_names.extend(map(lambda id_: ".".join(self.global_names[id_].split(".")[:-1]), functions))
            candidate_names = sorted(candidate_names, key=len)  # shortest name is assumed to be correct

            if len(candidate_names) > 0:
                return self.nodes.query(f"serialized_name == '{candidate_names[0]}' and type == 'module'").iloc[0]["id"]
            return None

        func_global, module_cand = get_global_id_and_module_candidates(func_nodes, ["function_name", "global_mention_rev"])
        new_node_ids.update(func_global)
        module_candidates.extend(module_cand)
        class_global, module_cand = get_global_id_and_module_candidates(class_nodes, ["class_name", "global_mention_rev"])
        new_node_ids.update(class_global)
        module_candidates.extend(module_cand)

        module_global = get_global_module_id(module_nodes, module_candidates, func_global, class_global)
        new_node_ids.update(module_global)

        return new_node_ids

    @staticmethod
    def merge_global_references(global_references, module_global_references):
        existing_len = len(global_references)
        to_add_len = len(module_global_references)
        global_references.update(module_global_references)
        new_len = len(global_references)
        assert new_len == existing_len + to_add_len


class ReplacementNodeResolver(NodeResolver):
    def __init__(self, nodes):

        self.nodeid2name = dict(zip(nodes['id'].tolist(), nodes['serialized_name'].tolist()))
        self.nodeid2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        self.valid_new_node = nodes['id'].max() + 1
        self.node_ids = {}
        self.new_nodes = []
        self.stashed_nodes = []

        self.old_nodes = nodes.copy()
        self.old_nodes['mentioned_in'] = pd.NA
        self.old_nodes = self.old_nodes.astype({'mentioned_in': 'Int32'})

    def stash_new_nodes(self):
        self.stashed_nodes.extend(self.new_nodes)
        self.new_nodes = []

    def resolve_substrings(self, node, replacement2srctrl):

        decorated = "@" in node.name
        assert not decorated

        name_ = copy(node.name)

        replacements = dict()
        global_node_id = []
        global_name = []
        global_type = []
        for name in re.finditer("srctrlrpl_[0-9]+", name_):
            if isinstance(name, re.Match):
                name = name.group()
            elif isinstance(name, str):
                pass
            else:
                print("Unknown type")
            if name.startswith("srctrlrpl_"):
                node_id = replacement2srctrl[name]["srctrl_id"]
                if type(node_id) is int:
                    global_node_id.append(node_id)
                    global_name.append(self.nodeid2name[node_id])
                    global_type.append(self.nodeid2type[node_id])
                else:
                    global_node_id.extend(node_id)
                    global_name.extend([self.nodeid2name[nid] for nid in node_id])
                    global_type.extend([self.nodeid2type[nid] for nid in node_id])
                replacements[name] = {
                    "name": replacement2srctrl[name]["original_string"],
                    "id": node_id
                }

        real_name = name_
        for r, v in replacements.items():
            real_name = real_name.replace(r, v["name"])

        return GNode(
            name=real_name, type=node.type, global_name=global_name, global_id=global_node_id, global_type=global_type
        )

    def resolve_regular_replacement(self, node, replacement2srctrl):

        decorated = "@" in node.name
        assert len([c for c in node.name if c == "@"]) <= 1

        if decorated:
            name_, decorator = node.name.split("@")
        else:
            name_, decorator = copy(node.name), None

        if name_ in replacement2srctrl:

            global_node_id = replacement2srctrl[name_]["srctrl_id"]
            # the only types that should appear here are "Name", "mention", "#attr#"
            # the first two are from mentions, and the last one is when references sourcetrail node is an attribute
            # if node.type not in {"Name", "mention", "#attr#"}:
            if node.type in {"#keyword#"}:
                # TODO
                # either a sourcetrail error or the error parsing
                return GNode(name=replacement2srctrl[name_]["original_string"], type=node.type)
            assert node.type in {"Name", "mention", "#attr#"}
            real_name = replacement2srctrl[name_]["original_string"]
            global_name = self.nodeid2name[global_node_id] if type(global_node_id) is int else\
                [self.nodeid2name[nid] for nid in global_node_id]
            global_type = self.nodeid2type[global_node_id] if type(global_node_id) is int else\
                [self.nodeid2type[nid] for nid in global_node_id]

            if node.type == "Name":
                # name always go together with mention, therefore no global reference in Name
                new_node = GNode(name=real_name, type=node.type)
            else:
                if decorated:
                    assert node.type == "mention"
                    real_name += "@" + decorator
                    type_ = "mention"
                else:
                    assert node.type == "#attr#"
                    type_ = node.type
                new_node = GNode(name=real_name, type=type_, global_name=global_name, global_id=global_node_id,
                                 global_type=global_type)
        else:
            new_node = node
        return new_node

    def resolve(self, node, replacement2srctrl):

        if node.type == "type_annotation":
            new_node = self.resolve_substrings(node, replacement2srctrl)
        else:
            new_node = self.resolve_regular_replacement(node, replacement2srctrl)
            if "srctrlrpl_" in new_node.name:  # hack to process imports
                new_node = self.resolve_substrings(node, replacement2srctrl)

        assert "srctrlrpl_" not in new_node.name

        return new_node

    def resolve_node_id(self, node, **kwargs):
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
                        "mentioned_in": pd.NA
                    }
                )
                if hasattr(node, "scope"):
                    self.resolve_node_id(node.scope)
                    self.new_nodes[-1]["mentioned_in"] = node.scope.id
                node.setprop("id", new_id)
        return node

    def prepare_for_write(self, from_stashed=False):
        nodes = pd.concat([self.old_nodes, self.new_nodes_for_write(from_stashed)])[
            ['id', 'type', 'serialized_name', 'mentioned_in']
        ]

        return nodes

    def new_nodes_for_write(self, from_stashed=False):

        new_nodes = pd.DataFrame(self.new_nodes if not from_stashed else self.stashed_nodes)[
            ['id', 'type', 'serialized_name', 'mentioned_in']
        ].astype({"mentioned_in": "Int32"})

        return new_nodes

    def adjust_ast_node_types(self, mapping, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["type"] = mapping.get(node["type"], node["type"])

    def drop_nodes(self, node_ids_to_drop, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        position = 0
        while position < len(nodes):
            if nodes[position]["id"] in node_ids_to_drop:
                nodes.pop(position)
            else:
                position += 1

    def map_mentioned_in_to_global(self, mapping, from_stashed=False):
        nodes = self.new_nodes if not from_stashed else self.stashed_nodes

        for node in nodes:
            node["mentioned_in"] = mapping.get(node["mentioned_in"], node["mentioned_in"])



class OffsetIndex:
    class IndexNode:
        def __init__(self, data):
            n_entries = len(data)
            self.start = data["start"].min()
            self.end = data["end"].max()
            if n_entries > 1:
                middle = n_entries // 2
                self.left = self.__class__(data.iloc[:middle])
                self.right = self.__class__(data.iloc[middle:])
            elif n_entries == 1:
                self.id = data["node_id"].iloc[0]
                self.occ_type = data["occ_type"].iloc[0]
            else:
                raise Exception("Invalid state")

        def get_overlapped(self, range, overlapped=None):
            if overlapped is None:
                overlapped = []
            if (self.start - range[0]) * (self.end - range[1]) <= 0:
                if hasattr(self, "id"):
                    overlapped.append((self.start, self.end, self.id, self.occ_type))
                else:
                    self.right.get_overlapped(range, overlapped)
                    self.left.get_overlapped(range, overlapped)
            return overlapped

    def __init__(self, data: pd.DataFrame):
        # data = data[["start", "end", "node_id"]].sort_values(by=["start", "end"])
        data = data.sort_values(by=["start", "end"])
        self.index = self.IndexNode(data)

    def get_overlap(self, range):
        return self.index.get_overlapped(range)


class AstProcessor(AstGraphGenerator):
    def get_edges(self, as_dataframe=True):
        edges = []
        all_edges, top_node_name = self.parse(self.root)
        edges.extend(all_edges)

        if as_dataframe:
            df = pd.DataFrame(edges)
            df = df.astype({col: "Int32" for col in df.columns if col not in {"src", "dst", "type"}})

            body = "\n".join(self.source)
            cum_lens = get_cum_lens(body, as_bytes=True)
            byte2char = get_byte_to_char_map(body)

            def format_offsets(edges: pd.DataFrame):
                edges["start_line__end_line__start_column__end_column"] = list(
                    zip(edges["line"], edges["end_line"], edges["col_offset"], edges["end_col_offset"])
                )

                def into_offset(range):
                    try:
                        return to_offsets(body, [(*range, None)], cum_lens=cum_lens, b2c=byte2char, as_bytes=True)[-1][:2]
                    except:
                        return None

                edges["offsets"] = edges["start_line__end_line__start_column__end_column"].apply(into_offset)
                edges.drop(
                    axis=1,
                    labels=[
                        "start_line__end_line__start_column__end_column",
                        "line",
                        "end_line",
                        "col_offset",
                        "end_col_offset"
                    ], inplace=True
                )

            format_offsets(df)
            return df
        else:
            body = "\n".join(self.source)
            cum_lens = get_cum_lens(body, as_bytes=True)
            byte2char = get_byte_to_char_map(body)

            def format_offsets(edge):
                def into_offset(range):
                    try:
                        return to_offsets(body, [(*range, None)], cum_lens=cum_lens, b2c=byte2char, as_bytes=True)[-1][:2]
                    except:
                        return None

                if "line" in edge:
                    edge["offsets"] = into_offset(
                        (edge["line"], edge["end_line"], edge["col_offset"], edge["end_col_offset"])
                    )
                    edge.pop("line")
                    edge.pop("end_line")
                    edge.pop("col_offset")
                    edge.pop("end_col_offset")
                else:
                    edge["offsets"] = None
                if "var_line" in edge:
                    edge["var_offsets"] = into_offset(
                        (edge["var_line"], edge["var_end_line"], edge["var_col_offset"], edge["var_end_col_offset"])
                    )
                    edge.pop("var_line")
                    edge.pop("var_end_line")
                    edge.pop("var_col_offset")
                    edge.pop("var_end_col_offset")

            for edge in edges:
                format_offsets(edge)

            return edges


class SourcetrailResolver:
    def __init__(self, nodes, edges, source_location, occurrence, file_content, lang):
        self.nodes = nodes
        self.node2name = dict(zip(nodes['id'], nodes['serialized_name']))

        self.edges = edges
        self.source_location = source_location
        self.occurrence = occurrence
        self.file_content = file_content
        self.lang = lang
        self._occurrence_groups = None

    @property
    def occurrence_groups(self):
        if self._occurrence_groups is None:
            self._occurrence_groups = get_occurrence_groups(self.nodes, self.edges, self.source_location, self.occurrence)
        return self._occurrence_groups

    def get_node_id_from_occurrence(self, elem_id__target_id__name):
        element_id, target_node_id, name = elem_id__target_id__name

        if not isinstance(name, str):
            node_id = target_node_id
        else:
            node_id = element_id

        assert node_id in self.node2name

        if self.node2name[node_id] == UNRESOLVED_SYMBOL:
            # this is an unresolved symbol, avoid
            return pd.NA
        else:
            return node_id

    def get_file_content(self, file_id):
        return self.file_content.query(f"id == {file_id}").iloc[0]['content']

    def occurrences_into_ranges(self, body, occurrences: pd.DataFrame):

        columns = ["element_id", "start_line", "end_line", "start_column", "end_column", "occ_type",
                   "target_node_id", "serialized_name"]
        cmap = {c: columns.index(c) for c in columns}
        occurrences = occurrences[columns].values

        new_occurrences = []
        for occurrence in occurrences:
            referenced_node = self.get_node_id_from_occurrence((
                occurrence[cmap["element_id"]], occurrence[cmap["target_node_id"]], occurrence[cmap["serialized_name"]]
            ))
            if pd.isna(referenced_node):
                continue
            name = self.node2name[referenced_node]
            info = {"node_id": referenced_node, "occ_type": occurrence[cmap["occ_type"]], "name": name}
            offset = (
                occurrence[cmap["start_line"]] - 1,
                occurrence[cmap["end_line"]] - 1,
                occurrence[cmap["start_column"]] - 1,
                occurrence[cmap["end_column"]], info
            )
            new_occurrences.append(offset)

        return self.offsets2dataframe(to_offsets(body, new_occurrences, as_bytes=True))

    @staticmethod
    def offsets2dataframe(offsets):
        records = []

        for offset in offsets:
            entry = {"start": offset[0], "end": offset[1]}
            entry.update(offset[2])
            records.append(entry)

        return pd.DataFrame(records)


def global_mention_edges_from_node(node):
    global_edges = []
    if type(node.global_id) is int:
        id_type = [(node.global_id, node.global_type)]
    else:
        id_type = zip(node.global_id, node.global_type)

    for gid, gtype in id_type:
        global_mention = {
            "src": GNode(name=None, type=gtype, id=gid),
            "dst": node,
            "type": "global_mention",
            "offsets": None
        }
        global_edges.append(global_mention)
        global_edges.append(make_reverse_edge(global_mention))
    return global_edges

def add_global_mentions(edges):
    new_edges = []
    for edge in edges:
        if edge['src'].type in {"#attr#", "Name"}:
            if hasattr(edge['src'], "global_id"):
                new_edges.extend(global_mention_edges_from_node(edge['src']))
        elif edge['dst'].type == "mention":
            if hasattr(edge['dst'], "global_id"):
                new_edges.extend(global_mention_edges_from_node(edge['dst']))
        new_edges.append(edge)
    return new_edges


def edges_for_global_node_names(nodes):
    edges = []
    for id, name, type in nodes[["id", "serialized_name", "type"]].values:
        # edges.append({
        #     "src": GNode(type="Name", name=name),
        #     "dst": GNode(id=id, type="__global", name=""),
        #     "type": "__global_name"
        # })
        mention = GNode(type="mention", name=f"{name}@{name}_0x")
        edges.append({
            "src": GNode(type="Name", name=name),
            "dst": mention,
            "type": "local_mention"
        })
        edges.append({
            "src": mention,
            "dst": GNode(id=id, type="__global", name=""),
            "type": type + "_name"
        })
    return edges


def produce_nodes_without_name(global_nodes, ast_edges):
    # from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types
    # global_types = set(list(node_types.values()))
    global_node_ids = set(global_nodes["id"].tolist())
    name_edge_types = {"function_name", "class_name"}

    global_nodes_with_name = set([edge["src"] for edge in ast_edges if edge["src"] in global_node_ids and edge["type"] in name_edge_types])
    nodes_without_name = global_nodes.query("id not in @global_nodes_with_name", local_dict={"global_nodes_with_name": global_nodes_with_name})
    
    return nodes_without_name


def standardize_new_edges(edges, node_resolver, mention_tokenizer):

    edges = mention_tokenizer.replace_mentions_with_subwords(edges)

    resolve_node_id = lambda node: node_resolver.resolve_node_id(node)
    extract_id = lambda node: node.id

    for edge in edges:
        edge["src"] = resolve_node_id(edge["src"])
        edge["dst"] = resolve_node_id(edge["dst"])
        if "scope" in edge:
            edge["scope"] = resolve_node_id(edge["scope"])

    for edge in edges:
        edge["src"] = extract_id(edge["src"])
        edge["dst"] = extract_id(edge["dst"])
        if "scope" in edge:
            edge["scope"] = extract_id(edge["scope"])

    return edges


def process_code(source_file_content, offsets, node_resolver, mention_tokenizer, node_matcher, track_offsets=False):
    replacer = OccurrenceReplacer()
    replacer.perform_replacements(source_file_content, offsets)

    ast_processor = AstProcessor(replacer.source_with_replacements)
    edges = ast_processor.get_edges(as_dataframe=False)

    if len(edges) == 0:
        return None, None, None

    resolve = lambda node: node_resolver.resolve(node, replacer.replacement_index)

    for edge in edges:
        edge["src"] = resolve(edge["src"])
        edge["dst"] = resolve(edge["dst"])
        if "scope" in edge:
            edge["scope"] = resolve(edge["scope"])

    edges = add_global_mentions(edges)

    edges = standardize_new_edges(edges, node_resolver, mention_tokenizer)

    if track_offsets:
        def get_valid_offsets(edges):
            return [(edge["offsets"][0], edge["offsets"][1], edge["src"]) for edge in edges if edge["offsets"] is not None]

        ast_offsets = replacer.recover_offsets_with_edits2(get_valid_offsets(edges))

        def merge_global_and_ast_offsets(ast_offsets, global_offsets):
            ast_offsets.extend(global_offsets)
            return ast_offsets

        global_and_ast_offsets = merge_global_and_ast_offsets(
            ast_offsets, global_offsets=offsets.query("occ_type != 1")[["start", "end", "node_id"]].values
        )
    else:
        global_and_ast_offsets = None

    ast_nodes_to_srctrl_nodes = node_matcher.match_with_global_nodes(node_resolver.new_nodes, edges)

    return edges, global_and_ast_offsets, ast_nodes_to_srctrl_nodes


def get_ast_from_modules(
        nodes, edges, source_location, occurrence, file_content,
        bpe_tokenizer_path, create_subword_instances, connect_subwords, lang
):

    srctrl_resolver = SourcetrailResolver(nodes, edges, source_location, occurrence, file_content, lang)
    node_resolver = ReplacementNodeResolver(nodes)
    node_matcher = GlobalNodeMatcher(nodes, add_reverse_edges(edges))
    mention_tokenizer = MentionTokenizer(bpe_tokenizer_path, create_subword_instances, connect_subwords)
    all_ast_edges = []
    all_global_references = {}
    all_offsets = []

    for group_ind, (file_id, occurrences) in custom_tqdm(
            enumerate(srctrl_resolver.occurrence_groups), message="Processing modules",
            total=len(srctrl_resolver.occurrence_groups)
    ):
        source_file_content = srctrl_resolver.get_file_content(file_id)

        if not has_valid_syntax(source_file_content):
            continue

        offsets = srctrl_resolver.occurrences_into_ranges(source_file_content, occurrences)

        # process code
        try:
            edges, global_and_ast_offsets, ast_nodes_to_srctrl_nodes = process_code(
                source_file_content, offsets, node_resolver, mention_tokenizer, node_matcher
            )
        except SyntaxError:
            logging.warning(f"Error processing file_id {file_id}")
            continue

        if edges is None:
            continue

        # afterprocessing

        for edge in edges:
            edge["file_id"] = file_id

        # finish afterprocessing

        all_ast_edges.extend(edges)
        node_matcher.merge_global_references(all_global_references, ast_nodes_to_srctrl_nodes)

        def format_offsets(global_and_ast_offsets, target):
            if global_and_ast_offsets is not None:
                for offset in global_and_ast_offsets:
                    target.append({
                        "file_id": file_id,
                        "start": offset[0],
                        "end": offset[1],
                        "node_id": offset[2]
                    })

        format_offsets(global_and_ast_offsets, target=all_offsets)

        node_resolver.stash_new_nodes()

    def replace_ast_node_to_global(edges, mapping):
        for edge in edges:
            edge["src"] = mapping.get(edge["src"], edge["src"])
            edge["dst"] = mapping.get(edge["dst"], edge["dst"])
            if "scope" in edge:
                edge["scope"] = mapping.get(edge["scope"], edge["scope"])

    replace_ast_node_to_global(all_ast_edges, all_global_references)

    def create_subwords_for_global_nodes():
        all_ast_edges.extend(
            standardize_new_edges(edges_for_global_node_names(produce_nodes_without_name(nodes, all_ast_edges)),
                                  node_resolver, mention_tokenizer))
        node_resolver.stash_new_nodes()

    create_subwords_for_global_nodes()

    def prepare_new_nodes(node_resolver):

        node_resolver.adjust_ast_node_types(
            mapping={
                "Module": "module",
                "FunctionDef": "function",
                "ClassDef": "class",
                "class_method": "function"
            }, from_stashed=True
        )
        node_resolver.map_mentioned_in_to_global(all_global_references, from_stashed=True)
        node_resolver.drop_nodes(set(all_global_references.keys()), from_stashed=True)

    prepare_new_nodes(node_resolver)

    all_ast_nodes = node_resolver.new_nodes_for_write(from_stashed=True)

    def prepare_edges(all_ast_edges):
        all_ast_edges = pd.DataFrame(all_ast_edges)
        all_ast_edges.drop_duplicates(["type", "src", "dst"], inplace=True)
        all_ast_edges = all_ast_edges.query("src != dst")
        all_ast_edges["id"] = 0
        all_ast_edges = all_ast_edges[["id", "type", "src", "dst", "file_id", "scope"]].rename({'src': 'source_node_id', 'dst': 'target_node_id', 'scope': 'mentioned_in'}, axis=1).astype(
            {'file_id': 'Int32', "mentioned_in": 'Int32'}
        )
        return all_ast_edges

    all_ast_edges = prepare_edges(all_ast_edges)

    if len(all_offsets) > 0:
        all_offsets = pd.DataFrame(all_offsets)
    else:
        all_offsets = None

    return all_ast_nodes, all_ast_edges, all_offsets


class OccurrenceReplacer:
    def __init__(self):
        self.replacement_index = None
        self.original_source = None
        self.source_with_replacements = None
        self.processed = None
        self.evicted = None
        self.edits = None

    @staticmethod
    def format_offsets_for_replacements(offsets):
        offsets = offsets.sort_values(by=["start", "end"], ascending=[True, False])
        return list(zip(offsets["start"], offsets["end"], list(zip(offsets["node_id"], offsets["occ_type"]))))

    @staticmethod
    def place_temp_to_evicted(temp_evicted, temp_end_changes, current_offset, evicted, source_code):
        pos = 0
        while pos < len(temp_evicted):
            if temp_evicted[pos][1] - temp_end_changes[pos] < current_offset[0]:
                temp_offset = temp_evicted.pop(pos)
                end_change = temp_end_changes.pop(pos)
                start = temp_offset[0]
                end = temp_offset[1] + end_change
                evicted.append(
                    {"start": start, "end": end, "sourcetrail_id": temp_offset[2][0],
                     "occ_type": temp_offset[2][1], "str": source_code[start: end]})
            else:
                pos += 1

    @staticmethod
    def group_overlapping_offsets(offset, pending):
        offsets = [offset]
        while len(pending) > 0 and range_overlap(offset, pending[0]):
            offsets.append(pending.pop(0))

        offsets = sorted(offsets, key=lambda x: x[1] - x[0])  # sort by offset span size

        # choose the smallest span
        offset = (offsets[0][0], offsets[0][1], list(set(o[2] for o in offsets)))
        if len(offset[2]) == 1:
            offset = (offset[0], offset[1], offset[2][0])
        return offset

    def perform_replacements(self, source_file_content, offsets):

        self.original_source = source_file_content

        pending = self.format_offsets_for_replacements(offsets)
        temp_evicted = []
        temp_end_changes = []
        processed = []
        evicted = []
        replacement_index = {}
        edits = []

        while len(pending) > 0:
            offset = pending.pop(0)  # format (start, end, (node_id, occ_type))

            self.place_temp_to_evicted(temp_evicted, temp_end_changes, offset, evicted, source_file_content)

            src_str = source_file_content[offset[0]: offset[1]]
            # longer occurrences such as attributes and function definition will be found first because occurences are
            # sorted by occurrence end position in descending order
            if ("." in src_str or "\n" in src_str or " " in src_str or
                    "[" in src_str or "(" in src_str or "{" in src_str):
                temp_evicted.append(offset)
                temp_end_changes.append(0)
            else:
                offset = self.group_overlapping_offsets(offset, pending)

                # new_name = f"srctrlnd_{offset[2][0]}"
                replacement_id = int(time_ns())
                new_name = "srctrlrpl_" + str(replacement_id)
                replacement_index[new_name] = {
                    "srctrl_id": offset[2][0] if type(offset[2]) is not list else [o[0] for o in offset[2]],
                    "original_string": src_str
                }
                old_len = offset[1] - offset[0]
                new_len = len(new_name)
                len_diff = new_len - old_len
                pending = adjust_offsets2(pending, len_diff)
                processed.append({"start": offset[0], "end": offset[1] + len_diff, "replacement_id": replacement_id})
                temp_end_changes = [val + len_diff for val in temp_end_changes]
                edits.append((offset[1], len_diff))
                source_file_content = source_file_content[:offset[0]] + new_name + source_file_content[offset[1]:]

        if len(temp_evicted) > 0:
            final_position = max(map(lambda x: x[0][1] + x[1], zip(temp_evicted, temp_end_changes)))
            self.place_temp_to_evicted(temp_evicted, temp_end_changes, (final_position,), evicted, source_file_content)

        self.source_with_replacements = source_file_content
        self.processed = pd.DataFrame(processed)
        self.evicted = pd.DataFrame(evicted)
        self.replacement_index = replacement_index
        self.edits = edits

    def recover_offsets_with_edits(self, offsets):
        pending = sorted(offsets, key=lambda x: x[0], reverse=True)
        edits = copy(self.edits)
        edits.reverse()

        recovered = []
        compound = []
        if len(edits) == 0:
            cum_adds = 0
            edit_position, edit_len = 0, 0
        else:
            cum_adds = sum(map(lambda x: x[1], edits))
            edit_position, edit_len = edits.pop(0)

        if len(pending) == 0:
            return []

        start, end, node_id = pending.pop(0)
        while len(pending) > 0:
            if start > edit_position + edit_len:
                recovered.append((start - cum_adds, end - cum_adds, node_id))
                start, end, node_id = pending.pop(0)
            elif end >= edit_position + edit_len:
                adjust = edit_len
                recovered.append((start - cum_adds + adjust, end - cum_adds, node_id))
                start, end, node_id = pending.pop(0)
            elif end < edit_position + edit_len:
                cum_adds -= edit_len
                edit_position, edit_len = edits.pop(0)
            else:
                raise Exception("Illegal scenario")

        return recovered

    def recover_offsets_with_edits2(self, offsets):
        # pending = copy(offsets)
        pending = sorted(offsets, key=lambda x: x[0], reverse=True)
        edits = copy(self.edits)
        edits.reverse()

        if len(edits) == 0:
            return pending

        if len(pending) == 0:
            return []

        for edit_position, edit_len in edits:
            tolerance = 20
            new_pending = []
            for ind, offset in enumerate(pending):
                start, end, node_id = offset
                needs_replacement = False
                if edit_len > 0:
                    if start > edit_position:
                        start = start - edit_len
                        needs_replacement = True
                    if end > edit_position:
                        end = end - edit_len
                        needs_replacement = True
                    if needs_replacement:
                        pending[ind] = (start, end, node_id)
                    else:
                        tolerance -= 1
                else:
                    if start > edit_position + edit_len:
                        start = start - edit_len
                        needs_replacement = True
                    if end >= edit_position + edit_len:
                        end = end - edit_len
                        needs_replacement = True
                    if needs_replacement:
                        pending[ind] = (start, end, node_id)
                    else:
                        tolerance -= 1

                if tolerance == 0:
                    break

                # if node_id == 261384:
                #     print()

        return pending


pd.options.mode.chained_assignment = None  # default='warn'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('working_directory', type=str,
                        help='Path to ')
    parser.add_argument('--bpe_tokenizer', '-bpe', dest='bpe_tokenizer', type=str,
                        help='')
    parser.add_argument('--create_subword_instances', action='store_true', default=False, help="")
    parser.add_argument('--connect_subwords', action='store_true', default=False,
                        help="Takes effect only when `create_subword_instances` is False")
    parser.add_argument('--lang', dest='lang', default="python", help="")

    args = parser.parse_args()

    working_directory = args.working_directory

    source_location = read_source_location(working_directory)
    occurrence = read_occurrence(working_directory)
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)
    file_content = read_filecontent(working_directory)

    ast_nodes, ast_edges, offsets = get_ast_from_modules(nodes, edges, source_location, occurrence, file_content,
                                                         args.bpe_tokenizer, args.create_subword_instances,
                                                         args.connect_subwords, args.lang)

    edges_with_ast_name = os.path.join(working_directory, "edges_with_ast.bz2")
    nodes_with_ast_name = os.path.join(working_directory, "nodes_with_ast.bz2")
    offsets_path = os.path.join(working_directory, "ast_offsets.bz2")

    persist(nodes.append(ast_nodes), nodes_with_ast_name)
    persist(edges.append(ast_edges), edges_with_ast_name)
    if offsets is not None:
        persist(offsets, offsets_path)