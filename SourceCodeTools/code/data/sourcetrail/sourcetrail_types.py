
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
    1: "defines",
    8: "calls",
    2: "uses_type",
    16: "inheritance",
    4: "uses",
    512: "imports"
}