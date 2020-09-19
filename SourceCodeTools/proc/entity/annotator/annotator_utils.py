def get_cum_lens(body):
    """
    Calculate the cummulative lengths of each line with respect to the beginning of
    the function's body.
    """
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line) + cum_lens[-1] + 1) # +1 for new line character
    return cum_lens

def to_offsets(body, entities):
    """
    Transform entity annotation format from (line, end_line, col, end_col)
    to (char_ind, end_char_ind).
    """
    cum_lens = get_cum_lens(body)

    repl = [(cum_lens[line] + start, cum_lens[end_line] + end, annotation) for
            ind, (line, end_line, start, end, annotation) in enumerate(entities)]

    return repl