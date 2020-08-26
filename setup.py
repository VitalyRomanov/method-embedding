from distutils.core import setup

setup(name='SourceCodeTools',
      version='0.0.1',
      py_modules=['SourceCodeTools'],
      install_requires=[
            'pandas',
            'javac_parser'
      ],
      scripts=[
            'SourceCodeTools/data/sourcetrail/sourcetrail-call-seq-extractor.py',
            'SourceCodeTools/data/sourcetrail/compress_to_bz2.py',
            'SourceCodeTools/data/sourcetrail/edge_types_to_int.py',
            'SourceCodeTools/data/sourcetrail/extract_node_names.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-extract-variable-names.py',
            'SourceCodeTools/data/sourcetrail/filter_ambiguous_edges.py',
            'SourceCodeTools/data/sourcetrail/filter_edges_by_type.py',
            'SourceCodeTools/data/sourcetrail/filter_packages.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-ast-edges.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-merge-graphs.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-parse-bodies.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-extract-docstring.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-edges-name-resolve.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-filter-edges.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-graph-properties-spark.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-names-to-packages.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-node-name-merge.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-verify-files.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-merge-graphs.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-map-id-columns.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-map-id-columns-only-annotations.py',
            'SourceCodeTools/data/sourcetrail/sourcetrail-node-local2global.py',
            'SourceCodeTools/proc/entity/annotator/sourcecodetools-extract-type-annotations.py',
            'SourceCodeTools/proc/entity/sourcecodetools-replace-tokenizer.py',
            'SourceCodeTools/proc/entity/sourcecodetools-spacy-ner.py',
            'SourceCodeTools/embed/ft_bin_to_vec.py',
      ],
)