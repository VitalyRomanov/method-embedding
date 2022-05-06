using LightGraphs, MetaGraphs


function add_new_edge!(graph, edge, type)
    node_id1 = graph.metaindex[:name][edge[1]]
    node_id2 = graph.metaindex[:name][edge[2]]

    e = Edge(node_id1, node_id2)

    if has_edge(graph, e)
        c_count = get_prop(graph, e, :count)
        set_prop!(graph, e, :count, c_count+1)
    else
        add_edge!(graph, node_id1, node_id2, Dict(:type => type, :count => 1))
    end
end

function add_node!(graph, node_name)
    if nv(graph) > 0
        node_id = get(graph.metaindex[:name], node_name, -1)
    else
        node_id = -1
    end
    if node_id == -1
        add_vertex!(graph)
        set_indexing_prop!(graph, nv(graph), :name, node_name)
    end
end




function read_graph(edge_list)
    graph = MetaDiGraph()

    for (ind, line) in enumerate(eachline(edge_list))

        if (ind == 1)
            continue
        end

        if length(strip(line)) > 0
            parts = split(strip(line), ",")
            if length(parts) == 4
                src, dst = parts[3], parts[4]

                add_node!(graph, src)
                add_node!(graph, dst)

                add_new_edge!(graph, (src, dst), parts[2])

            end
        end

    end
    graph
end

function generate_walks(graph, sink, MAX_PATH_LEN=15)
    n_paths = nv(graph)

    get_name(node_id) = begin graph[node_id, :name] end

    for p in 1:n_paths
        seed_vertex = rand(vertices(graph))

        walk = randomwalk(graph, seed_vertex, MAX_PATH_LEN)

        if length(walk) < 2
            continue
        end

        nodes = get_name.(walk)

        write(sink, "$(join(nodes, "\t"))\n")

        if p % 1000 == 0
            print("\r$(p)/$(n_paths)")
        end
    end
    println()
end

cd("/Volumes/External/datasets/Code/source-graphs/python-source-graph/component_0")
EDGE_LIST_PATH = "edges_component_0.csv"
edge_list = open(EDGE_LIST_PATH, "r")
sink = open("walks_jl.txt", "w")

graph = read_graph(edge_list)
generate_walks(graph, sink)
