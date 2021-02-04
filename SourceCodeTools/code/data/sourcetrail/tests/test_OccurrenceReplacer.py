

def test_OccurrenceReplacer():
    from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import OccurrenceReplacer, AstProcessor
    import pandas as pd


    # test_case = """class TestStrCategoryFormatter:
    #     test_cases = [("ascii", ["hello", "world", "hi"]),
    #                   ("unicode", ["Здравствуйте", "привет"])]
    #     ids, cases = zip(*test_cases)"""
    #
    # offsets = pd.DataFrame([
    #     {"start": 6, "end": 30, "node_id": 1, "occ_type": 0},
    #     {"start": 40, "end": 50, "node_id": 2, "occ_type": 0},
    #     {"start": 175, "end": 178, "node_id": 3, "occ_type": 0},
    #     {"start": 180, "end": 190, "node_id": 4, "occ_type": 0},
    # ])

    # test_case = """class TestStrCategoryFormatterTestStrCategoryFormatterTestStrCategoryFormatterTestStrCategoryFormatter:
    #     test_cases = [("ascii", ["hello", "world", "hi"]),
    #                   ("unicode", ["Здравствуйте", "привет"])]
    #     ids, cases = zipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzip(*test_cases)"""
    #
    test_case = """class TestStrCategoryFormatterTestStrCategoryFormatterTestStrCategoryFormatterTestStrCategoryFormatter:
        test_cases = [("ascii", ["hello", "world", "hi"]),
                      ("unicode", ["Здравствуйте", "привет"])]
        ids, cases = zipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzipzip(*test_cases)"""

    offsets = pd.DataFrame([
        {"start": 6, "end": 102, "node_id": 1, "occ_type": 0},
        {"start": 112, "end": 122, "node_id": 2, "occ_type": 0},
        {"start": 247, "end": 319, "node_id": 3, "occ_type": 0},
        {"start": 321, "end": 331, "node_id": 4, "occ_type": 0},
    ])


#     test_case = """#-----------------------------------------------------------------------------
# # Copyright (c) 2012 - 2020, Anaconda, Inc., and Bokeh Contributors.
# # All rights reserved.
# #
# # The full license is in the file LICENSE.txt, distributed with this software.
# #-----------------------------------------------------------------------------
# ''' Define a Pytest plugin for a Bokeh-specific testing tools
# '''
# #-----------------------------------------------------------------------------
# # Boilerplate
# #-----------------------------------------------------------------------------
# import logging # isort:skip
# log = logging.getLogger(__name__)
# #-----------------------------------------------------------------------------"""
#
#     offsets = pd.DataFrame([
#         {"start": 605, "end": 612, "node_id": 9, "occ_type": 0},
#         {"start": 599, "end": 602, "node_id": 11, "occ_type": 0},
#         {"start": 578, "end": 585, "node_id": 9, "occ_type": 0},
#         {"start": 613, "end": 622, "node_id": 13, "occ_type": 0},
#     ])

    replacer = OccurrenceReplacer()
    replacer.perform_replacements(test_case, offsets)

    ast_processor = AstProcessor(replacer.source_with_replacements)
    edges = ast_processor.get_edges(as_dataframe=False)

    def get_valid_offsets(edges):
        return [(edge["offsets"][0], edge["offsets"][1], edge["src"]) for edge in edges if edge["offsets"] is not None]

    replaced_offsets = get_valid_offsets(edges)

    ast_offsets = replacer.recover_offsets_with_edits(replaced_offsets)

    for offset in replaced_offsets:
        print(f"{offset[2].name}\t{replacer.source_with_replacements[offset[0]:offset[1]]}")

    print("\n\n\n")

    for offset in ast_offsets:
        print(f"{offset[2].name}\t{replacer.original_source[offset[0]:offset[1]]}")

    print()

# test_OccurrenceReplacer()

"""class srctrlrpl_1612301920884920000:
        srctrlrpl_1612301920884960000 = [("ascii", ["hello", "world", "hi"]),
                      ("unicode", ["zzzzzzzzzzzz", "zzzzzz"])]
        ids, cases = srctrlrpl_1612301920884984000(*srctrlrpl_1612301920885006000)
"""


def test_OccurrenceReplacer_gcrp():
    from SourceCodeTools.code.data.sourcetrail.sourcetrail_ast_edges2 import OccurrenceReplacer, AstProcessor
    import pandas as pd
    from gcrp_test_case import gcrp_offsets, gcrp_test_string

    replacer = OccurrenceReplacer()
    replacer.perform_replacements(gcrp_test_string, pd.DataFrame(gcrp_offsets))

    ast_processor = AstProcessor(replacer.source_with_replacements)
    edges = ast_processor.get_edges(as_dataframe=False)

test_OccurrenceReplacer_gcrp()