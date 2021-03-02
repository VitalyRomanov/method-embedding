from SourceCodeTools.nlp.entity import parse_biluo


def test_parse_biluo():
    assert parse_biluo(["O", "U-1", "0", "B-2", "L-2", "O", "O", "B-3", "I-3", "L-3"]) == [(1, 2, '1'), (3, 5, '2'), (7, 10, '3')]
    assert parse_biluo(["O", "U-1", "0", "B-2", "L-2", "O", "O", "B-3", "B-3", "L-3"]) == [(1, 2, '1'), (3, 5, '2')]
    assert parse_biluo(["O", "U-1", "0", "B-2", "U-2", "O", "O", "B-3", "I-3", "L-3"]) == [(1, 2, '1'), (7, 10, '3')]