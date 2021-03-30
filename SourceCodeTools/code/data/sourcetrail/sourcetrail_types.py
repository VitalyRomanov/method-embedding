
node_types = {
    128: "class",
    4096: "function",
    2048: "class_field",
    8: "module",
    8192: "class_method",
    1024: "global_variable",
    1: "non_indexed_symbol",
}

edge_types = {
    1: "defines",       # from module to function
    8: "calls",         # from caller to callee
    2: "uses_type",     # from user to type
    16: "inheritance",
    4: "uses",          # from user to item
    512: "imports"      # from module to imported object
}


special_mapping = {
        "defines": "defined_in",
        "calls": "called_by",
        "uses_type": "type_used_by",
        "inheritance": "inherited_by",
        "uses": "used_by",
        "imports": "imported_by"
    }