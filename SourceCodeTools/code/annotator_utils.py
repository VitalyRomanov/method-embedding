from copy import copy, deepcopy
from typing import List, Tuple, Iterable

from SourceCodeTools.nlp import create_tokenizer
from spacy.gold import biluo_tags_from_offsets as spacy_biluo_tags_from_offsets

from SourceCodeTools.nlp.tokenizers import codebert_to_spacy


def biluo_tags_from_offsets(doc, ents, no_localization):
    ent_tags = spacy_biluo_tags_from_offsets(doc, ents)

    if no_localization:
        tags = []
        for ent in ent_tags:
            parts = ent.split("-")

            assert len(parts) <= 2

            if len(parts) == 2:
                if parts[0] == "B" or parts[0] == "U":
                    tags.append(parts[1])
                else:
                    tags.append("O")
            else:
                tags.append("O")

        ent_tags = tags

    return ent_tags


def get_cum_lens(body, as_bytes=False):
    """
    Calculate the cummulative lengths of each line with respect to the beginning of
    the function's body.
    """
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line if not as_bytes else line.encode('utf8')) + cum_lens[-1] + 1) # +1 for new line character
    return cum_lens


from SourceCodeTools.nlp.string_tools import get_byte_to_char_map
# def get_byte_to_char_map(unicode_string):
#     """
#     Generates a dictionary mapping character offsets to byte offsets for unicode_string.
#     """
#     response = {}
#     byte_offset = 0
#     for char_offset, character in enumerate(unicode_string):
#         response[byte_offset] = char_offset
#         # print(character, byte_offset, char_offset)
#         byte_offset += len(character.encode('utf-8'))
#     response[byte_offset] = len(unicode_string)
#     return response


def to_offsets(body: str, entities: Iterable[Iterable], as_bytes=False, cum_lens=None, b2c=None):
    """
    Transform entity annotation format from (line, end_line, col, end_col)
    to (char_ind, end_char_ind).
    :param body: string containing function body
    :param entities: list of tuples containing entity start- and end-offsets in bytes
    :param as_bytes: treat entity offsets as offsets for bytes. this is needed when offsets are given in bytes, not in str positions
    :return: list of tuples that represent start- and end-offsets in a string that contains function body
    """
    if cum_lens is None:
        cum_lens = get_cum_lens(body, as_bytes=as_bytes)

    # b2c = [get_byte_to_char_map(line) for line in body.split("\n")]

    # repl = [(cum_lens[line] + b2c[line][start], cum_lens[end_line] + b2c[end_line][end], annotation) for
    #         ind, (line, end_line, start, end, annotation) in enumerate(entities)]
    repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
            ind, (line, end_line, start, end, annotation) in enumerate(entities)]

    if as_bytes:
        if b2c is None:
            b2c = get_byte_to_char_map(body)
        repl = list(map(lambda x: (b2c[x[0]], b2c[x[1]], x[2]), repl))

    return repl


def adjust_offsets(offsets, amount):
    """
    Adjust offset by subtracting certain amount from the start and end positions
    :param offsets: iterable with offsets
    :param amount: adjustment amount
    :return: list of adjusted offsets
    """
    if amount == 0:
        return offsets
    return [(offset[0] - amount, offset[1] - amount, offset[2]) for offset in offsets]


def adjust_offsets2(offsets, amount):
    """
    Adjust offset by adding certain amount to the start and end positions
    :param offsets: iterable with offsets
    :param amount: adjustment amount
    :return: list of adjusted offsets
    """
    return [(offset[0] + amount, offset[1] + amount, offset[2]) for offset in offsets]


def overlap(p1: Tuple, p2: Tuple) -> bool:
    """
    Check whether two entities defined by (start_position, end_position) overlap
    :param p1: tuple for the first entity
    :param p2: tuple for the second entity
    :return: boolean flag whether two entities overlap
    """
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False


def resolve_self_collision(offsets):
    no_collisions = []

    for ind_1, offset_1 in enumerate(offsets):
        # keep first
        if any(map(lambda x: overlap(offset_1, x), no_collisions)):
            pass
        else:
            no_collisions.append(offset_1)
        # new = []
        # evict = []
        #
        # for ind_2, offset_2 in enumerate(no_collisions):
        #     if overlap(offset_1, offset_2):
        #         # keep smallest
        #         if (offset_1[1] - offset_1[0]) <= (offset_2[1] - offset_2[0]):
        #             evict.append(ind_2)
        #             new.append(offset_1)
        #         else:
        #             pass
        #     else:
        #         new.append(offset_1)
        #
        # for ind in sorted(evict, reverse=True):
        #     no_collisions.pop(ind)
        #
        # no_collisions.extend(new)

    return no_collisions


def resolve_self_collisions2(offsets):
    """
    Resolve self collision in favour of the smallest entity.
    :param offsets:
    :return:
    """
    offsets = copy(offsets)
    no_collisions = []

    while len(offsets) > 0:
        offset_1 = offsets.pop(0)
        evict = []
        new = []

        add = True
        for ind_2, offset_2 in enumerate(no_collisions):
            if overlap(offset_1, offset_2):
                # keep smallest
                if (offset_1[1] - offset_1[0]) <= (offset_2[1] - offset_2[0]):
                    evict.append(ind_2)
                    new.append(offset_1)
                else:
                    pass
                add = False

        if add:
            new.append(offset_1)

        for ind in sorted(evict, reverse=True):
            no_collisions.pop(ind)

        no_collisions.extend(new)

    no_collisions = list(set(no_collisions))

    return no_collisions


def align_tokens_with_graph(doc, spans, tokenzer_name):
    spans = deepcopy(spans)
    if tokenzer_name == "codebert":
        backup_tokens = doc
        doc, adjustment = codebert_to_spacy(doc)
        spans = adjust_offsets(spans, adjustment)

    node_tags = biluo_tags_from_offsets(doc, spans, no_localization=False)

    if tokenzer_name == "codebert":
        doc = ["<s>"] + [t.text for t in backup_tokens] + ["</s>"]
    return doc, node_tags


def source_code_graph_alignment(source_codes, node_spans, tokenizer="codebert"):
    supported_tokenizers = ["spacy", "codebert"]
    assert tokenizer in supported_tokenizers, f"Only these tokenizers supported for alignment: {supported_tokenizers}"
    nlp = create_tokenizer(tokenizer)

    for code, spans in zip(source_codes, node_spans):
        yield align_tokens_with_graph(nlp(code), resolve_self_collisions2(spans), tokenzer_name=tokenizer)


def map_offsets(column, id_map):
    def map_entry(entry):
        return [(e[0], e[1], id_map[e[2]]) for e in entry]
    return [map_entry(entry) for entry in column]