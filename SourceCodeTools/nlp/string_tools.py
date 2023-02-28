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


def get_byte_to_char_map2(unicode_string):
    """
    Generates a dictionary mapping character offsets to byte offsets for unicode_string.
    """
    response = {}
    byte_offset = 0
    for char_offset, character in enumerate(unicode_string):
        bytes = character.encode('utf-8')
        for byte_ind, _ in enumerate(bytes):
            response[byte_offset + byte_ind] = char_offset
        # print(character, byte_offset, char_offset)
        byte_offset += len(bytes)
    response[byte_offset] = len(unicode_string)
    return response
