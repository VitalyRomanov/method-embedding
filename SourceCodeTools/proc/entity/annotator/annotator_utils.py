from typing import List, Tuple


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


def get_byte_to_char_map(unicode_string):
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    """
    response = {}
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        response[byte_offset] = char_offset
        # print(character, byte_offset, char_offset)
        byte_offset += len(character.encode('utf-8'))
    response[byte_offset] = len(unicode_string)
    return response


def to_offsets(body: str, entities: List[Tuple], as_bytes=False):
    """
    Transform entity annotation format from (line, end_line, col, end_col)
    to (char_ind, end_char_ind).
    :param body: string containing function body
    :param entities: list of tuples containing entity start- and end-offsets in bytes
    :param as_bytes: treat entity offsets as offsets for bytes. this is needed when body contains non-ascii characters
    :return: list of tuples that represent start- and end-offsets in a string that contains function body
    """
    cum_lens = get_cum_lens(body, as_bytes=as_bytes)

    # b2c = [get_byte_to_char_map(line) for line in body.split("\n")]

    # repl = [(cum_lens[line] + b2c[line][start], cum_lens[end_line] + b2c[end_line][end], annotation) for
    #         ind, (line, end_line, start, end, annotation) in enumerate(entities)]
    repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
            ind, (line, end_line, start, end, annotation) in enumerate(entities)]

    if as_bytes:
        b2c = get_byte_to_char_map(body)
        repl = list(map(lambda x: (b2c[x[0]], b2c[x[1]], x[2]), repl))

    return repl


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