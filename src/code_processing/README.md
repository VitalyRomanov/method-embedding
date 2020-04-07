Collection of Python scripts for extracting additional information from function bodies stored in Sourcetrail database.

#### `call_seq_extractor.py` 
Usage: `call_seq_extractor.py path/to/sourcetrail/export` 
Creates `consequent_calls.txt`, where each line is a sequence of functions called inside another funtion

#### `extract_variable_names.py` 
Usage: `extract_variable_names.py path/to/sourcetrail/functions path/to/sourcetrail/bodies` 
Creates `variable_name_ids.txt`, where each line encodes a unique variable name with an ID. Creates `function_variable_pairs.txt` that stores edges between functions and variable ID that are used inside a function. 
