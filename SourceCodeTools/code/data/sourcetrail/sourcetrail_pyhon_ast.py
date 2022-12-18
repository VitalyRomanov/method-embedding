from collections import defaultdict

from SourceCodeTools.code.ast.python_ast2 import PythonNodeEdgeDefinitions, PythonAstGraphBuilder
from SourceCodeTools.code.ast.python_ast2_with_subwords import PythonAstWithSubwordsEdgeDefinitions, \
    PythonAstWithSubwordsGraphBuilder
from SourceCodeTools.code.data.sourcetrail.common import *
from SourceCodeTools.code.data.sourcetrail.sourcetrail_types import node_types


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
                self.name = data["type"].iloc[0]
                self.type = data["name"].iloc[0]
                self.occ_type = data["occ_type"].iloc[0]
            else:
                raise Exception("Invalid state")

        def get_overlapped(self, range, overlapped=None):
            if overlapped is None:
                overlapped = []
            if (self.start - range[0]) * (self.end - range[1]) <= 0:
                if hasattr(self, "id"):
                    overlapped.append((self.start, self.end, (self.id, self.name, self.type)))
                else:
                    self.right.get_overlapped(range, overlapped)
                    self.left.get_overlapped(range, overlapped)
            return overlapped

    def __init__(self, data: pd.DataFrame):
        data = data.sort_values(by=["start", "end"])
        self.index = self.IndexNode(data)

    def get_overlap(self, range):
        return self.index.get_overlapped(range)


class SourcetrailPythonAstEdgeDefinitions(PythonNodeEdgeDefinitions):

    extra_node_types = PythonNodeEdgeDefinitions.extra_node_types | set(
        node_types.values()
    )

    extra_edge_types = PythonNodeEdgeDefinitions.extra_edge_types | {
        "global_mention"
    }

    reverse_edge_exceptions = {
        **PythonNodeEdgeDefinitions.reverse_edge_exceptions, **{
            "global_mention": None,
        }
    }


class SourcetrailPythonAstWithSubwordsEdgeDefinitions(PythonAstWithSubwordsEdgeDefinitions):

    extra_node_types = PythonNodeEdgeDefinitions.extra_node_types | set(
        node_types.values()
    )

    extra_edge_types = PythonNodeEdgeDefinitions.extra_edge_types | {
        "global_mention"
    }

    reverse_edge_exceptions = {
        **PythonNodeEdgeDefinitions.reverse_edge_exceptions, **{
            "global_mention": None,
        }
    }


class SourcetrailPythonAstGraphBuilder(PythonAstGraphBuilder):
    def __init__(
            self, *args, global_offsets=None, **kwargs
    ):
        super(SourcetrailPythonAstGraphBuilder, self).__init__(*args, **kwargs)
        self._global_offsets = global_offsets

    def postprocess(self):
        super().postprocess()
        self._attach_global_nodes()

    def _attach_global_nodes(self):
        global_offset_index = OffsetIndex(self._global_offsets)

        detected_overlaps = defaultdict(list)
        all_edges = list(self._edges)
        for edge in all_edges:
            node = self._node_pool[edge.src]
            if node.type in {
                self._node_types["ClassDef"],
                self._node_types["FunctionDef"]
            }:
                continue
            if edge.positions is not None:
                overlaps = global_offset_index.get_overlap(edge.positions)
                for o in overlaps:
                    detected_overlaps[o].append((*edge.positions, node))

        for globals, locals in detected_overlaps.items():
            overlap_scores = [
                ((min(globals[1], local[1])-max(globals[0], local[0]))-(local[1]-local[0]))/(globals[1]-globals[0]) for local in locals
            ]
            top_match = sorted(zip(locals, overlap_scores), key=lambda x:x[1], reverse=True)[0][0][2]
            global_node = self._get_node(name=globals[2][2], type=self._node_types[globals[2][1]])
            self._add_edge(
                self._edges, src=global_node, dst=top_match.hash_id, type=self._edge_types["global_mention"],
                scope=top_match.scope, position=(globals[0], globals[1])
            )


class SourcetrailPythonAstWithSubwordsGraphBuilder(PythonAstWithSubwordsGraphBuilder):
    def __init__(
            self, *args, global_offsets=None, **kwargs
    ):
        super(SourcetrailPythonAstWithSubwordsGraphBuilder, self).__init__(*args, **kwargs)
        self._global_offsets = global_offsets

    def postprocess(self):
        super().postprocess()
        self._attach_global_nodes()

    def _attach_global_nodes(self):
        global_offset_index = OffsetIndex(self._global_offsets)

        detected_overlaps = defaultdict(list)
        all_edges = list(self._edges)
        for edge in all_edges:
            node = self._node_pool[edge.src]
            if node.type in {
                self._node_types["ClassDef"],
                self._node_types["FunctionDef"]
            }:
                continue
            if edge.positions is not None:
                overlaps = global_offset_index.get_overlap(edge.positions)
                for o in overlaps:
                    detected_overlaps[o].append((*edge.positions, node))

        for globals, locals in detected_overlaps.items():
            overlap_scores = [
                ((min(globals[1], local[1])-max(globals[0], local[0]))-(local[1]-local[0]))/(globals[1]-globals[0]) for local in locals
            ]
            top_match = sorted(zip(locals, overlap_scores), key=lambda x:x[1], reverse=True)[0][0][2]
            global_node = self._get_node(name=globals[2][2], type=self._node_types[globals[2][1]])
            self._add_edge(
                self._edges, src=global_node, dst=top_match.hash_id, type=self._edge_types["global_mention"],
                scope=top_match.scope, position=(globals[0], globals[1])
            )


def make_sourcetrail_python_ast_graph(source_code, global_offsets, add_reverse_edges=False, save_node_strings=False):
    g = SourcetrailPythonAstGraphBuilder(
        source_code, SourcetrailPythonAstEdgeDefinitions, global_offsets=global_offsets, add_reverse_edges=add_reverse_edges, save_node_strings=save_node_strings
    )
    return g.to_df()


def make_sourcetrail_python_ast_with_subword_graph(source_code, global_offsets, add_reverse_edges=False, save_node_strings=False):
    g = SourcetrailPythonAstWithSubwordsGraphBuilder(
        source_code, SourcetrailPythonAstWithSubwordsEdgeDefinitions, global_offsets=global_offsets, add_reverse_edges=add_reverse_edges, save_node_strings=save_node_strings
    )
    return g.to_df()