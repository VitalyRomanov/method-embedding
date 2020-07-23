Collection of Python scripts for processing Sourcetrail graph.

#### `sourcetrail-node-name-merge.py` 
Usage: `sourcetrail-node-name-merge.py sourcetrail_nodes.csv` 
Creates `normalized_sourcetrail_nodes.csv`, where node names are processed in human-readable format. 

#### `sourcetrail-edges-name-resolve.py`
Usage: `sourcetrail-node-name-merge.py normalized_nodes.csv sourcetrail_edges.csv` 
Creates `normalized_sourcetrail_edges.csv` node ids in edgs are replaced with node names.

#### `sourcetrail_extract_docstring.py`
Usage: `sourcetrail_extract_docstring.py path/to/exported/csv`
Creates jsonl file. Each entry epcifies funtion id and docstring itself.

#### `filter_ambiguous_edges.py`
Usage: `filter_ambiguous_edges.py path/to/exported/csv` 
Creates two file: `ambiguous_edges.csv` and `non-ambiguous_edges.csv`.

#### `sourcetrail-graph-properties-spark.py`
Usage: `sourcetrail-graph-properties-spark.py nodes edges`
Does some analysis of graph properties, applies desired filtering.

#### `query_graph.py`
Returns immediate neighbours of a cetrain node.

#### `compress_edges.py`
Replaces node names in the list of edges with node ids.

#### `call_seq_extractor.py` 
Usage: `call_seq_extractor.py path/to/sourcetrail/export` 
Creates `consequent_calls.txt`, where each line is a sequence of functions called inside another funtion

#### `extract_variable_names.py` 
Usage: `extract_variable_names.py path/to/sourcetrail/functions path/to/sourcetrail/bodies` 
Creates `variable_name_ids.txt`, where each line encodes a unique variable name with an ID. Creates `function_variable_pairs.txt` that stores edges between functions and variable ID that are used inside a function. 
