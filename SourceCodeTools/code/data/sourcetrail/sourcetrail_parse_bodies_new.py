import argparse
import ast
import random
import string
import sys
from copy import copy
from time import time_ns
from typing import Tuple, List, Optional
import re

from SourceCodeTools.code.data.sourcetrail.common import *
from SourceCodeTools.nlp.entity.annotator.annotator_utils import adjust_offsets2

from SourceCodeTools.code.python_ast import AstGraphGenerator, GNode, PythonSharedNodes
from SourceCodeTools.nlp.entity.annotator.annotator_utils import overlap as range_overlap

from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges import NodeResolver, \
    produce_subword_edges_with_instances, produce_subword_edges, global_mention_edges_from_node, make_reverse_edge


class MentionTokenizer:
    def __init__(self, bpe_tokenizer_path, create_subword_instances, connect_subwords):
        from SourceCodeTools.nlp.embed.bpe import make_tokenizer
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model

        self.bpe = make_tokenizer(load_bpe_model((bpe_tokenizer_path))) \
            if bpe_tokenizer_path else None
        self.create_subword_instances = create_subword_instances
        self.connect_subwords = connect_subwords

    def replace_mentions_with_subwords(self, edges):
        edges = edges.to_dict(orient="records")

        if self.create_subword_instances:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges_with_instances(subwords, dst)
        else:
            def produce_subw_edges(subwords, dst):
                return self.produce_subword_edges(subwords, dst, self.connect_subwords)

        new_edges = []
        for edge in edges:
            if edge['src'].type in {"#attr#", "Name"}:
                if hasattr(edge['src'], "global_id"):
                    new_edges.extend(self.global_mention_edges_from_node(edge['src']))
            elif edge['dst'].type == "mention":
                if hasattr(edge['dst'], "global_id"):
                    new_edges.extend(self.global_mention_edges_from_node(edge['dst']))

            if edge['type'] == "local_mention":
                # is_global_mention = hasattr(edge['src'], "id")
                # if is_global_mention:
                #     # this edge connects sourcetrail node need to add couple of links
                #     # to ensure global information flow
                #     new_edges.extend(global_mention_edges(edge))

                dst = edge['dst']

                if self.bpe is not None:
                    if hasattr(dst, "name_scope") and dst.name_scope == "local":
                        subwords = self.bpe(dst.name.split("@")[0])
                    else:
                        subwords = self.bpe(edge['src'].name)

                    new_edges.extend(produce_subw_edges(subwords, dst))
                else:
                    new_edges.append(edge)

            elif self.bpe is not None and \
                    (
                            edge['src'].type in PythonSharedNodes.tokenizable_types
                    ) or (
                    edge['dst'].type in {"Global"} and edge['src'].type != "Constant"
            ):
                new_edges.append(edge)
                new_edges.append(make_reverse_edge(edge))

                dst = edge['src']
                subwords = self.bpe(dst.name)
                new_edges.extend(produce_subw_edges(subwords, dst))
            else:
                new_edges.append(edge)

        return pd.DataFrame(new_edges)

    def global_mention_edges_from_node(self, node):
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

    @staticmethod
    def produce_subword_edges(subwords, dst, connect_subwords=False):
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
                if ind < len(subwords) - 1:
                    new_edges.append({
                        'src': subword,
                        'dst': subwords[ind + 1],
                        'type': 'next_subword',
                        'offsets': None
                    })
                if ind > 0:
                    new_edges.append({
                        'src': subword,
                        'dst': subwords[ind - 1],
                        'type': 'prev_subword',
                        'offsets': None
                    })

        return new_edges

    @staticmethod
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
                'offsets': None
            })
            new_edges.append({
                'src': subword_instance,
                'dst': dst,
                'type': 'subword',
                'offsets': None
            })
            if ind < len(subwords) - 1:
                new_edges.append({
                    'src': subword_instance,
                    'dst': instances[ind + 1],
                    'type': 'next_subword',
                    'offsets': None
                })
            if ind > 0:
                new_edges.append({
                    'src': subword_instance,
                    'dst': instances[ind - 1],
                    'type': 'prev_subword',
                    'offsets': None
                })

        return new_edges


class ReplacementNodeResolver(NodeResolver):
    def __init__(self, nodes):

        self.nodeid2name = dict(zip(nodes['id'].tolist(), nodes['serialized_name'].tolist()))
        self.nodeid2type = dict(zip(nodes['id'].tolist(), nodes['type'].tolist()))

        self.valid_new_node = nodes['id'].max() + 1
        self.node_ids = {}
        self.new_nodes = []

        self.old_nodes = nodes.copy()

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

        return GNode(name=real_name, type=node.type, global_name=global_name, global_id=global_node_id, global_type=global_type)

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
            global_name = self.nodeid2name[global_node_id] if type(global_node_id) is int else [self.nodeid2name[nid] for
                                                                                                nid in global_node_id]
            global_type = self.nodeid2type[global_node_id] if type(global_node_id) is int else [self.nodeid2type[nid] for
                                                                                                nid in global_node_id]

            if node.type == "Name":
                # name always go together with mention, therefore no global reference in Name
                new_node = GNode(name=real_name, type=node.type, global_id=global_node_id, global_type=global_type)
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
            if "@" not in new_node.name and new_node.name == node.name:  # hack to process imports
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
                        "serialized_name": node.name
                    }
                )
                node.setprop("id", new_id)
        return node


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
    def get_edges(self):
        edges = []
        all_edges, top_node_name = self.parse(self.root)
        edges.extend(all_edges)

        df = pd.DataFrame(edges)
        df = df.astype({col: "Int32" for col in df.columns if col not in {"src", "dst", "type"}})

        body = "\n".join(self.source)
        cum_lens = get_cum_lens(body)
        def format_offsets(edges: pd.DataFrame):
            edges["start_line__end_line__start_column__end_column"] = list(zip(edges["line"], edges["end_line"], edges["col_offset"], edges["end_col_offset"]))
            def into_offset(range):
                try:
                    return to_offsets(body, [(*range, None)], cum_lens=cum_lens)[-1][:2]
                except:
                    return None
            edges["offsets"] = edges["start_line__end_line__start_column__end_column"].apply(into_offset)
            edges.drop(axis=1, labels=["start_line__end_line__start_column__end_column", "line", "end_line", "col_offset", "end_col_offset"], inplace=True)

        format_offsets(df)
        return df


from SourceCodeTools.nlp.entity.annotator.annotator_utils import to_offsets, get_cum_lens

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
            self._occurrence_groups = get_occurrence_groups(nodes, edges, source_location, occurrence)
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
        return file_content.query(f"id == {file_id}").iloc[0]['content']

    def occurrences_into_ranges(self, body, occurrences: pd.DataFrame):

        occurrences = occurrences.copy()
        occurrences["temp"] = list(zip(occurrences["element_id"], occurrences["target_node_id"], occurrences["serialized_name"]))
        occurrences["referenced_node"] = occurrences["temp"].apply(self.get_node_id_from_occurrence)
        occurrences.dropna(axis=0, subset=["referenced_node"], inplace=True)

        occurrences = occurrences[["start_line", "end_line", "start_column", "end_column", "referenced_node", "occ_type"]]
        occurrences['names'] = occurrences['referenced_node'].apply(lambda id_: self.node2name[id_])
        occurrences['elem_id__occ_type'] = [{"node_id": e_id, "occ_type": o_type, "name": name} for e_id, o_type, name in zip(occurrences["referenced_node"], occurrences["occ_type"], occurrences['names'])]
        occurrences.drop(labels=["referenced_node", "occ_type", "names"], axis=1, inplace=True)
        occurrences["start_line"] = occurrences["start_line"] - 1
        occurrences["end_line"] = occurrences["end_line"] - 1
        occurrences["start_column"] = occurrences["start_column"] - 1
        return self.offsets2dataframe(to_offsets(body, occurrences.values, as_bytes=True))

    @staticmethod
    def offsets2dataframe(offsets):
        records = []

        for offset in offsets:
            entry = {"start": offset[0], "end": offset[1]}
            entry.update(offset[2])
            records.append(entry)

        return pd.DataFrame(records)

    def process_modules(self, bpe_tokenizer_path, create_subword_instances, connect_subwords):

        bodies = []

        node_resolver = ReplacementNodeResolver(self.nodes)
        mention_tokenizer = MentionTokenizer(bpe_tokenizer_path, create_subword_instances, connect_subwords)

        for group_ind, (file_id, occurrences) in custom_tqdm(
                enumerate(self.occurrence_groups), message="Processing function bodies",
                total=len(self.occurrence_groups)
        ):
            source_file_content = self.get_file_content(file_id)

            offsets = self.occurrences_into_ranges(source_file_content, occurrences)

            replacer = OccurrenceReplacer()
            replacer.perform_replacements(source_file_content, offsets)

            ast_processor = AstProcessor(replacer.source_with_replacements)
            edges = ast_processor.get_edges()

            if len(edges) == 0:
                continue

            resolve = lambda node: node_resolver.resolve(node, replacer.replacement_index)

            edges['src'] = edges['src'].apply(resolve)
            edges['dst'] = edges['dst'].apply(resolve)

            edges = mention_tokenizer.replace_mentions_with_subwords(edges)

            resolve_node_id = lambda node: node_resolver.resolve_node_id(node)

            edges['src'] = edges['src'].apply(resolve_node_id)
            edges['dst'] = edges['dst'].apply(resolve_node_id)

            extract_id = lambda node: node.id
            edges['src'] = edges['src'].apply(extract_id)
            edges['dst'] = edges['dst'].apply(extract_id)

            edges = edges.drop_duplicates(subset=["src", "dst", "type"])

            edges['id'] = 0

            nodes_with_mentions = edges[edges["offsets"].apply(lambda x: x is not None)]
            nodes_with_mentions["node_id__offset"] = list(zip(nodes_with_mentions["src"], nodes_with_mentions["offsets"]))
            nodes_with_mentions["node_id__offset"] = nodes_with_mentions["node_id__offset"].apply(lambda x: (x[1][0], x[1][1], x[0]))

            ast_offsets = replacer.recover_offsets_with_edits(nodes_with_mentions["node_id__offset"].values)

            # TODO
            #  resolve modules ind function definitions

            #######################################

            offsets_index = OffsetIndex(offsets)

            def join_srctrl_and_ast_offsets(range):
                if range is None:
                    return None
                else:
                    return offsets_index.get_overlap(range)

            ast_edges["srctrl_overlap"] = ast_edges["offsets"].apply(join_srctrl_and_ast_offsets)
            overlaps = []
            for item in ast_edges["srctrl_overlap"]:
                if item is None:
                    continue
                else:
                    overlaps.extend(item)
            unique_overlaps = set(overlaps)

            function_definitions = get_function_definitions(occurrences)

            if len(function_definitions):
                for ind, f_def in function_definitions.iterrows():
                    f_start = f_def.start_line
                    f_end = f_def.end_line

                    local_occurrences = get_occurrences_from_range(occurrences, start=f_start, end=f_end)

                    # move to zero-index
                    f_start -= 1
                    f_end -= 1

                    body = get_function_body(file_content, file_id, f_start, f_end)

                    if not has_valid_syntax(body):
                        continue

                    processed = process_body(body, local_occurrences, nodes, f_def.element_id, f_start)

                    if processed is not None:
                        bodies.append(processed)

            # print(f"\r{group_ind}/{len(occurrence_groups)}", end="")

        # print(" " * 30, end="\r")

        if len(bodies) > 0:
            bodies_processed = pd.DataFrame(bodies)
            return bodies_processed
        else:
            return None


class OccurrenceReplacer:
    def __init__(self):
        self.replacement_index = None
        self.original_source = None
        self.source_with_replacements = None
        self.processed = None
        self.evicted = None

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
                    {"start": start, "end": end, "sourcetrail_id": temp_offset[2][0], "occ_type": temp_offset[2][1], "str": source_code[start: end]})
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
            if "." in src_str or "\n" in src_str or " " in src_str or "[" in src_str or "(" in src_str or "{" in src_str:
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

        final_position = max(map(lambda x: x[0][1] + x[1], zip(temp_evicted, temp_end_changes)))
        self.place_temp_to_evicted(temp_evicted, temp_end_changes, (final_position,), evicted, source_file_content)

        self.source_with_replacements = source_file_content
        self.processed = pd.DataFrame(processed)
        self.evicted = pd.DataFrame(evicted)
        self.replacement_index = replacement_index
        self.edits = edits

    def recover_offsets_with_edits(self, offsets):
        pending = sorted(offsets, key=lambda x: x[0], reverse=True)
        pending = sorted(pending, key=lambda x: x[1], reverse=True)
        edits = copy(self.edits)
        edits.reverse()

        recovered = []
        compound = []
        if len(edits) == 0:
            cum_adds = 0
            edit_postion, edit_len = 0, 0
        else:
            cum_adds = sum(map(lambda x: x[1], edits))
            edit_postion, edit_len = edits.pop(0)

        if len(pending) == 0:
            return []

        offset = pending.pop(0)
        while len(pending) > 0:
            if len(pending) == 3:
                print()
            if len(edits) > 0 and offset[0] <= edit_postion < edit_postion + edit_len <= offset[1] and offset[0] <= edits[0][0] < edits[0][0] + edits[0][1] <= offset[1]:
                compound.append(edits.pop(0))
            elif edit_postion + edit_len <= offset[0]:
                recovered.append((offset[0] - cum_adds, offset[1] - cum_adds, offset[2]))
                offset = pending.pop(0)
            elif offset[0] <= edit_postion < edit_postion + edit_len <= offset[1]:
                adjust = edit_len
                for c in compound:
                    adjust += c[1]
                recovered.append((offset[0] - cum_adds + adjust, offset[1] - cum_adds, offset[2]))
                offset = pending.pop(0)
            elif offset[1] < edit_postion:
                cum_adds -= edit_len
                if len(compound) > 0:
                    edit_postion, edit_len = compound.pop(0)
                else:
                    edit_postion, edit_len = edits.pop(0)
            else:
                raise Exception("Illegal")

        return recovered






class RandomReplacementException(Exception):
    def __init__(self, message):
        super(RandomReplacementException, self).__init__(message)


pd.options.mode.chained_assignment = None  # default='warn'


def overlap(range: Tuple[int, int], ranges: List[Tuple[int, int]]) -> bool:
    for r in ranges:
        if (r[0] - range[0]) * (r[1] - range[1]) <= 0:
            return True
    return False


def isnamechar(char: str) -> bool:
    return "A" <= char <= "Z" or \
           "a" <= char <= "z" or \
           char == "." or \
           char == "_" or \
           "0" <= char <= "9"


def extend_range(start: int, end: int, line: str) -> Optional[Tuple[int, int]]:
    # assume only the following symbols are possible in names: A-Z a-z 0-9 . _
    # if start - 1 > 0 and line[start - 1] == "!":
    #     # used in f-strings
    #     # f"[{attr_selector!r}]"
    #     return None
    return start, end
    # if start - 1 > 0 and isnamechar(line[start-1]):
    #     return extend_range(start - 1, end, line)
    # elif start - 1 > 0 and line[start - 1] == "!":
    #     # used in f-strings
    #     # f"[{attr_selector!r}]"
    #     return None
    # else:
    #     if start - 1 > 0 and line[start] == "." and not isnamechar(line[start - 1]):
    #         return start + 1, end
    #     return start, end


def do_replacement(string_: str, start: int, end: int, substitution: str):
    return string_[:start] + substitution + \
                             string_[end:]


def get_docstring_ast(body):
    try:
        # this does not work with java, docstring formats are different
        ast_parse = ast.parse(body.strip())
        function_definitions = [node for node in ast_parse.body if isinstance(node, ast.FunctionDef)]
        return ast.get_docstring(function_definitions[0])
    except:
        return ""


def get_random_string(str_len):
    return "".join(random.choices(string.ascii_letters, k=str_len))


def generate_random_replacement(len: int, source: List[str]):
    attempts_left = 30
    secondary_attempts = 10
    replacement = get_random_string(len)
    body = "\n".join(source)
    while body.find(replacement) != -1:
        attempts_left -= 1
        replacement = get_random_string(len)
        if attempts_left <= 0:
            secondary_attempts -= 1
            replacement = "".join(random.choices("АБВГДЕЖЗИКЛМНОПРСТУФХЦЧЪЫЬЭЮЯабвгдежзиклмнопрстуфхцчъыьэюя", k=len))
            if secondary_attempts == 0:
                raise RandomReplacementException("Could not replace with random name")

    return replacement


def get_function_body(file_content, file_id, start, end) -> str:
    source_lines = file_content.query(f"id == {file_id}").iloc[0]['content'].split("\n")

    body_lines = source_lines[start: end]

    body_num_lines = len(body_lines)

    trim = 0

    if body_num_lines > 2: # assume one line function, or signaure and return statement
        initial_indent = len(body_lines[0]) - len(body_lines[0].lstrip())

        while trim < body_num_lines:
            # print("body_num_lines - 1 > 1", body_num_lines - 1 > 1)
            # print("""body_lines[body_num_lines - trim - 1].strip() == """"", body_lines[body_num_lines - trim - 1].strip() == "")
            # print("< < initial_indent", len(body_lines[body_num_lines - trim - 1]) - \
            #         len(body_lines[body_num_lines - trim - 1].lstrip()), initial_indent, len(body_lines[body_num_lines - trim - 1]) - \
            #         len(body_lines[body_num_lines - trim - 1].lstrip()) <= initial_indent)
            cline = body_lines[body_num_lines - trim - 1]
            if body_num_lines - 1 > 1:
                if cline.strip() == "":
                    trim += 1
                elif len(cline) - len(cline.lstrip()) <= initial_indent:
                    trim += 1
                else:
                    break
            else:
                break

        if trim == body_num_lines:
            trim = 0

    return "\n".join(source_lines[start: end - trim])


def has_valid_syntax(function_body):
    # body_lines = function_body.rstrip().split("\n")
    #
    # initial_indent = len(body_lines[0]) - len(body_lines[0].lstrip())
    # final_indent = len(body_lines[-1]) - len(body_lines[-1].lstrip())
    #
    # if body_lines[-1].strip() != "" and len(body_lines) > 1 and initial_indent >= final_indent:
    #     # will also skip functions like this:
    #     # '    def func(ax, x, y): pass
    #     #     def func_args(ax, x, y, *args): pass'
    #     # but this is probably not a big deal
    #     return False

    try:
        ast.parse(function_body.lstrip())
        return True
    except SyntaxError:
        return False


def get_range_for_replacement(occurrence, start_col, end_col, line, nodes):
    extended_range = extend_range(start_col, end_col, line)

    if extended_range is None:
        return None, None

    st_id = occurrence.element_id

    name = occurrence.serialized_name
    if not isinstance(name, str):  # happens when id refers to an edge, not a node
        st_id = occurrence.target_node_id

    node_info = nodes.query(f"id == {st_id}")
    assert len(node_info) == 1

    if node_info.iloc[0]['serialized_name'] == UNRESOLVED_SYMBOL:
        # this is an unresolved symbol, avoid
        return None, None
    else:
        name = f"srctrlnd_{st_id}"  # sourcetrailnode
        return extended_range, name


def get_occurrence_string(line, col_start, col_end):
    return line[col_start: col_end]


def _process_body(body, local_occurrences, nodes, f_id, f_start):
    body_normalized = body.split("\n")
    body_with_random_replacements = body.split("\n")
    random_2_original = {}
    random_2_srctrl = {}

    # assert source_lines[f_start] == body_with_random_replacements[0]

    local_occurrences = sort_occurrences(local_occurrences)

    prev_line = 0
    replaced_ranges = []
    list_of_replacements = []

    for occ_ind, occurrence in local_occurrences.iterrows():
        if occurrence.start_line == occurrence.end_line:

            curr_line = occurrence.start_line - 1 - f_start

            if curr_line >= len(body_normalized):
                continue

            if prev_line != curr_line:
                replaced_ranges = []
                # assert body_with_random_replacements[curr_line - f_start] == source_lines[curr_line]

            start_col = occurrence.start_column - 1
            end_col = occurrence.end_column

            if not overlap((start_col, end_col), replaced_ranges):
                extended_range, sourcetrail_name = get_range_for_replacement(
                    occurrence, start_col, end_col, body_normalized[curr_line], nodes
                )

                if extended_range is not None:
                    replaced_ranges.append(extended_range)
                    occ_col_start, occ_col_end = extended_range

                    list_of_replacements.append((
                        curr_line, occ_col_start, occ_col_end, sourcetrail_name
                    ))

                    random_name = generate_random_replacement(
                        len=occ_col_end - occ_col_start, source=body_with_random_replacements
                    )
                    random_2_original[random_name] = get_occurrence_string(
                        body_with_random_replacements[curr_line], occ_col_start, occ_col_end
                    )
                    body_with_random_replacements[curr_line] = do_replacement(
                        body_with_random_replacements[curr_line], occ_col_start, occ_col_end, random_name
                    )
                    random_2_srctrl[random_name] = sourcetrail_name

                    body_normalized[curr_line] = do_replacement(
                        body_normalized[curr_line], occ_col_start, occ_col_end, sourcetrail_name
                    )

            prev_line = curr_line

    norm_body = "\n".join(body_normalized)
    body_with_random_replacements = "\n".join(body_with_random_replacements)

    assert has_valid_syntax(norm_body)  # this should always be correct
    if not has_valid_syntax(body_with_random_replacements):  # this can fail
        raise RandomReplacementException("Syntax error after replacements")

    return {
        "id": f_id, #  f_def.element_id,
        "body": body,
        "body_normalized": norm_body,
        "body_with_random_replacements": body_with_random_replacements,
        "docstring": get_docstring_ast(body),
        "replacement_list": list_of_replacements,
        "random_2_original": random_2_original,
        "random_2_srctrl": random_2_srctrl
    }


def process_body(body, local_occurrences, nodes, f_id, f_start):
    replacement_attempts = 100

    while replacement_attempts > 0:
        try:
            return _process_body(body, local_occurrences, nodes, f_id, f_start)
        except RandomReplacementException:
            replacement_attempts -= 1

    return None






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
    # bodies_processed = process_modules(nodes, edges, source_location, occurrence, file_content, lang)
    srctrl_resolver = SourcetrailResolver(nodes, edges, source_location, occurrence, file_content, args.lang)
    bodies_processed = srctrl_resolver.process_modules(args.bpe_tokenizer, args.create_subword_instances, args.connect_subwords)
    if bodies_processed is not None:
        write_processed_bodies(bodies_processed, working_directory)