from distutils.core import setup

requitements = [
      'nltk',
      'tensorflow>=2.4.0',
      'torch>=1.7.1',
      'pandas>=1.1.1',
      'sklearn',
      'sentencepiece',
      'gensim',
      'numpy>=1.19.2',
      'scipy',
      'networkx',
      'sacrebleu',
      'datasets'
      # 'pygraphviz'
      # 'javac_parser'
]

setup(name='SourceCodeTools',
      version='0.0.2',
      py_modules=['SourceCodeTools'],
      install_requires=requitements + ["dgl"],
      extras_require={
            "gpu": requitements + ["dgl-cu110"]
      },
      scripts=[
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_call_seq_extractor.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_extract_node_names.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_extract_variable_names.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_filter_ambiguous_edges.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_compute_function_diameter.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_add_reverse_edges.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_ast_edges2.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_merge_graphs.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_parse_bodies2.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_edges_name_resolve.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_node_name_merge.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_decode_edge_types.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_verify_files.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_map_id_columns.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_map_id_columns_only_annotations.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_node_local2global.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_connected_component.py',
            'SourceCodeTools/code/data/sourcetrail/pandas_format_converter.py',
            'SourceCodeTools/code/data/sourcetrail/sourcetrail_create_type_annotation_dataset.py',
            'SourceCodeTools/nlp/embed/converters/convert_fasttext_format_bin_to_vec.py',
      ],
)
