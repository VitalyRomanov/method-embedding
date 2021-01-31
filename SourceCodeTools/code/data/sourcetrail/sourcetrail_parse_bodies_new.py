import ast
import random
import string
import sys
from time import time_ns
from typing import Tuple, List, Optional

from SourceCodeTools.code.data.sourcetrail.common import *
from SourceCodeTools.nlp.entity.annotator.annotator_utils import adjust_offsets2

from SourceCodeTools.code.python_ast import AstGraphGenerator
from SourceCodeTools.nlp.entity.annotator.annotator_utils import overlap as range_overlap


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

    def process_modules(self):

        bodies = []

        for group_ind, (file_id, occurrences) in custom_tqdm(
                enumerate(self.occurrence_groups), message="Processing function bodies",
                total=len(self.occurrence_groups)
        ):
            source_file_content = self.get_file_content(file_id)

            offsets = self.occurrences_into_ranges(source_file_content, occurrences)

            replacer = OccurrenceReplacer()
            replacer.perform_replacements(source_file_content, offsets)

            ast_processor = AstProcessor(replacer.source_with_replacements)
            ast_edges = ast_processor.get_edges()

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
                replacement_index[replacement_id] = {
                    "srctrl_id": offset[2][0] if type(offset[2]) is not list else [o[0] for o in offset[2]],
                    "original_string": src_str
                }
                old_len = offset[1] - offset[0]
                new_len = len(new_name)
                len_diff = new_len - old_len
                pending = adjust_offsets2(pending, len_diff)
                processed.append({"start": offset[0], "end": offset[1] + len_diff, "replacement_id": replacement_id})
                temp_end_changes = [val + len_diff for val in temp_end_changes]
                source_file_content = source_file_content[:offset[0]] + new_name + source_file_content[offset[1]:]

        final_position = max(map(lambda x: x[0][1] + x[1], zip(temp_evicted, temp_end_changes)))
        self.place_temp_to_evicted(temp_evicted, temp_end_changes, (final_position,), evicted, source_file_content)

        self.source_with_replacements = source_file_content
        self.processed = pd.DataFrame(processed)
        self.evicted = pd.DataFrame(evicted)
        self.replacement_index = replacement_index


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
    working_directory = sys.argv[1]
    try:
        lang = sys.argv[2]
    except:
        lang = "python"

    source_location = read_source_location(working_directory)
    occurrence = read_occurrence(working_directory)
    nodes = read_nodes(working_directory)
    edges = read_edges(working_directory)
    file_content = read_filecontent(working_directory)
    # bodies_processed = process_modules(nodes, edges, source_location, occurrence, file_content, lang)
    srctrl_resolver = SourcetrailResolver(nodes, edges, source_location, occurrence, file_content, lang)
    bodies_processed = srctrl_resolver.process_modules()
    if bodies_processed is not None:
        write_processed_bodies(bodies_processed, working_directory)