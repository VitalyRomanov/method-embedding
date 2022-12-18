import argparse

from SourceCodeTools.code.ast import has_valid_syntax
from SourceCodeTools.code.common import custom_tqdm
from SourceCodeTools.code.data.sourcetrail.common import *
from SourceCodeTools.code.annotator_utils import to_offsets
from SourceCodeTools.code.data.sourcetrail.sourcetrail_pyhon_ast import make_sourcetrail_python_ast_graph, \
    make_sourcetrail_python_ast_with_subword_graph


class SourcetrailResolver:
    """
    Helper class to work with source code stored in Sourcetrail database. Implements functions of
    - Iterating over files
    - Preserving Sourcetrail nodes
    """
    def __init__(self, nodes, edges, source_location, occurrence, file_content, lang):
        self.nodes = nodes
        self.node2name = dict(zip(nodes['id'], nodes['serialized_name']))
        self.node2type = dict(zip(nodes['id'], nodes['type']))

        self.edges = edges
        self.source_location = source_location
        self.occurrence = occurrence
        self.file_content = file_content
        self.lang = lang
        self._occurrence_groups = None

    @property
    def occurrence_groups(self):
        """
        :return: Iterator for occurrences grouped by file id.
        """
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
            type = self.node2type[referenced_node]
            info = {"node_id": referenced_node, "occ_type": occurrence[cmap["occ_type"]], "name": name, "type": type}
            offset = (
                occurrence[cmap["start_line"]] - 1,
                occurrence[cmap["end_line"]] - 1,
                occurrence[cmap["start_column"]] - 1,
                occurrence[cmap["end_column"]], info
            )
            new_occurrences.append(offset)

        return self.offsets2dataframe(to_offsets(body, new_occurrences))

    @staticmethod
    def offsets2dataframe(offsets):
        records = []

        for offset in offsets:
            entry = {"start": offset[0], "end": offset[1]}
            entry.update(offset[2])
            records.append(entry)

        return pd.DataFrame(records)


def get_ast_from_modules(
        nodes, edges, source_location, occurrence, file_content,
        bpe_tokenizer_path, create_subword_instances, connect_subwords, lang, track_offsets=False,
        package_name=None
):
    """
    Create edges from source code and methe them wit hthe global graph. Prepare all offsets in the uniform format.
    :param nodes: DataFrame with nodes
    :param edges: DataFrame with edges
    :param source_location: Dataframe that links nodes to source files
    :param occurrence: Dataframe with records about occurrences
    :param file_content: Dataframe with sources
    :param bpe_tokenizer_path: path to sentencepiece model
    :param create_subword_instances:
    :param connect_subwords:
    :param lang:
    :param track_offsets:
    :return: Tuple:
        - Dataframe with all nodes. Schema: id, type, name, mentioned_in (global and AST)
        - Dataframe with all edges. Schema: id, type, src, dst, file_id, mentioned_in
        - Dataframe with all_offsets. Schema: file_id, start, end, node_id, mentioned_in
    """
    srctrl_resolver = SourcetrailResolver(nodes, edges, source_location, occurrence, file_content, lang)

    file_ast = {}

    for group_ind, (file_id, occurrences) in custom_tqdm(
            enumerate(srctrl_resolver.occurrence_groups), message="Processing modules",
            total=len(srctrl_resolver.occurrence_groups)
    ):
        source_file_content = srctrl_resolver.get_file_content(file_id)

        if not has_valid_syntax(source_file_content):
            continue

        offsets = srctrl_resolver.occurrences_into_ranges(source_file_content, occurrences)
        offsets = offsets.query("occ_type != 1")

        nodes, edges = make_sourcetrail_python_ast_with_subword_graph(
            source_file_content, global_offsets=offsets, add_reverse_edges=True, save_node_strings=True
        )

        file_ast[file_id] = (source_file_content, nodes, edges)

    return file_ast


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