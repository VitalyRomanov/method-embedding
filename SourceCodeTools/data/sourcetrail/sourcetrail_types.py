
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
    1: "definition",
    8: "fcall",
    2: "type_use",
    16: "inheritance",
    4: "usage",
    512: "import"
}