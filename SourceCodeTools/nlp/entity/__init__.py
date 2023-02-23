def parse_biluo(biluo):
    """
    Parse BILUO and return token spans for entities
    :param biluo: list of BILUO tokens
    :return: list of token spans
    """
    spans = []

    expected = {"B", "U", "0"}
    expected_tag = None

    c_start = 0

    for ind, t in enumerate(biluo):
        if t[0] not in expected:
            # TODO
            # skips U-tag if it follows incorrect labels
            expected = {"B", "U", "0"}
            continue

        if t[0] == "U":
            c_start = ind
            c_end = ind + 1
            c_type = t.split("-")[1]
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "B":
            c_start = ind
            expected = {"I", "L"}
            expected_tag = t.split("-")[1]
        elif t[0] == "I":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
        elif t[0] == "L":
            if t.split("-")[1] != expected_tag:
                expected = {"B", "U", "0"}
                expected_tag = None
                continue
            c_end = ind + 1
            c_type = expected_tag
            spans.append((c_start, c_end, c_type))
            expected = {"B", "U", "0"}
            expected_tag = None
        elif t[0] == "0":
            expected = {"B", "U", "0"}
            expected_tag = None

    return spans


def tags_to_mask(tags):
    """
    Create a mask for BILUO tags where non-"O" tags are marked with 1.
    :param tags: list tags to mask
    :return: list of 1. and 0. that represent mask. For a list of input tags ['O', 'B-X', 'I-X', 'L-X', 'O'] it
    will return [0., 1., 1., 1., 0.]
    """
    return list(map(lambda t: 1. if t != "O" else 0., tags))


def fix_incorrect_tags(tags):
    while "-" in tags:
        tags[tags.index("-")] = "O"
