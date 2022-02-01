from distutils.core import setup

requirements = [
      'nltk==3.6',
      'tensorflow==2.6.0',
      'torch==1.9.0',
      'pandas==1.1.1',
      'scikit-learn==1.0',
      'sentencepiece==0.1.96',
      'gensim==3.8',
      'numpy==1.19.5',
      'scipy==1.4.1',
      'networkx==2.5',
      'sacrebleu==1.5.1',
      'datasets==1.5.0',
      'spacy==2.3.2',
      'pytest==6.1.2',
      'faiss-cpu==1.7.0'
      # 'pygraphviz'
      # 'javac_parser'
]
# conda install pytorch cudatoolkit=11.1 dgl-cuda11.1 -c dglteam -c pytorch -c nvidia
setup(name='SourceCodeTools',
      version='0.0.3',
      py_modules=['SourceCodeTools'],
      install_requires=requirements + ["dgl==0.7.1"],
      extras_require={
            "gpu": requirements + ["dgl-cu111==0.7.1"]
      },
      dependency_links=['https://data.dgl.ai/wheels/repo.html'],
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
            'SourceCodeTools/models/graph/utils/prepare_dglke_format.py',
      ],
      )
