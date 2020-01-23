nodes = dict()

with open("normalized_sourcetrail_nodes.csv", "r") as node_file:
    for ind, line in enumerate(node_file):
        if line.strip():
            if ind == 0: continue
            n_id, n_type, n_name = line.strip().split(",")
            if n_name in nodes:
                print("Duplicate:", n_name)
            nodes[n_name] = n_id

with open("normalized_sourcetrail_edges.csv", "r") as edges_file:
    with open("edges.csv", "w") as compressed_edges:
        for ind, line in enumerate(edges_file):
            if line.strip():
                if ind == 0: 
                    compressed_edges.write("%s\n" % line.strip())
                    continue
                e_id, e_type, e_s, e_d = line.strip().split(",")
                compressed_edges.write(f"{e_id},{e_type},{nodes[e_s]},{nodes[e_d]}\n")
            
            if ind % 10000 == 0:
                print("\r%d" % ind, end="")

