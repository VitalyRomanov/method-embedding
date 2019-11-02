using LightGraphs, MetaGraphs
using Dates
using JSON
using JLD
# include("LanguageTools.jl")

PARKING_LOT = "parking_lot.txt"
GRAPH = "graph.jld"

NORMAL_TYPE = "normal"
HAS_A_TYPE = "has_a"
PATTERN_NEIGH_TYPE = "pattern_neighbour"


# graph = MetaDiGraph()
# vertex_index = Dict()

# TODO
# 1. Need to capture the scope in which two items are pattern pattern_neighbours

struct NamedGraph
    mgraph
    index
    inv_index
end

function build_graph()

# en_lemm = LanguageTools.load("en_lemma.dict")

if isfile(PARKING_LOT)
    parking_lot = open(PARKING_LOT, "a")
else
    parking_lot = open(PARKING_LOT, "w")
end

if isfile(GRAPH)
    # graph = JLD.load(GRAPH, "graph")
    @load GRAPH graph
else
    graph = NamedGraph(MetaDiGraph(), Dict(), Dict())
    # graph = (mgraph=MetaDiGraph(), index=Dict(), inv_index=Dict())
end


function is_ambiguous(pattern)
    if length(pattern["super"]) > 1 || any(length(sub) > 1 for sub in pattern["sub"])
        return true
    end
    return false
end


function get_super(pattern)
    return pattern["super"]
end


function get_sub(pattern)
    return vcat(pattern["sub"]...)
end


function normalize(concept)
    # n_concept = join([LanguageTools.remove_accents(get(en_lemm, word, word)) for word in LanguageTools.tokenize(concept)], " ")
    return concept
end


function add_node!(graph, node_name)
    if node_name in keys(graph.index)
        node_id = graph.index[node_name]
        c_count = get_prop(graph.mgraph, node_id, :count)
        set_prop!(graph.mgraph, node_id, :count, c_count+1)
    else
        add_vertex!(graph.mgraph, Dict(:name => node_name, :count => 1))
        graph.index[node_name] = nv(graph.mgraph)
        graph.inv_index[nv(graph.mgraph)] = node_name
    end
end


function add_new_edge!(graph, edge, type)
    node_id1 = graph.index[edge[1]]
    node_id2 = graph.index[edge[2]]
    e = Edge(node_id1, node_id2)

    if has_edge(graph.mgraph, e)
        c_count = get_prop(graph.mgraph, e, :count)
        set_prop!(graph.mgraph, e, :count, c_count+1)
    else
        add_edge!(graph.mgraph, node_id1, node_id2, Dict(:type => type, :count => 1))
    end
end


count = 0

for line in eachline(stdin)

    if length(strip(line)) > 0
        pattern = JSON.parse(line)

        if is_ambiguous(pattern)
            # write(parking_lot, "$line\n")
            nothing
        else
            sup = get_super(pattern)[1]
            sup_normal_form = normalize(sup)

            add_node!(graph, sup)
            add_node!(graph, sup_normal_form)

            if sup != sup_normal_form
                add_new_edge!(graph, (sup, sup_normal_form), NORMAL_TYPE)
            end

            sub_c = get_sub(pattern)

            for concept in sub_c
                normal_form = normalize(concept)
                add_node!(graph, concept)
                add_node!(graph, normal_form)
                if concept != normal_form
                    add_new_edge!(graph, (concept, normal_form), NORMAL_TYPE)
                end
                add_new_edge!(graph, (sup_normal_form, normal_form), HAS_A_TYPE)
            end

            for concept in get_sub(pattern)
                normal_form = normalize(concept)

                for neigh in sub_c
                    neigh_normal_form = normalize(neigh)
                    if neigh_normal_form != normal_form 
                        add_new_edge!(graph, (normal_form, neigh_normal_form), PATTERN_NEIGH_TYPE)
                    end
                end
            end


        end
    end

    count = count + 1
    if count % 5000 == 0
        println("$(Dates.now()) Processed $count facts")
    end
end

# JLD.save(GRAPH, "graph", graph, compress=true)
close(parking_lot)
return graph
end
#@save GRAPH graph
# JLD.save(GRAPH, "graph", graph, compress=false)
# exit()

# for node in vertices(graph.mgraph)
#     println("$(get_prop(graph.mgraph, node, :name))\t$(get_prop(graph.mgraph, node, :count))")
# end

# for edge in edges(graph.mgraph)
#     n1 = get_prop(graph.mgraph, edge.src, :name)
#     n2 = get_prop(graph.mgraph, edge.dst, :name)
#     println("$n1\t$n2\t$(get_prop(graph.mgraph, edge, :count)) $(get_prop(graph.mgraph, edge, :type))")
# end
# exit()


# vertex_index = Dict()

# add_vertex!(graph, Dict(:count => 1))
# vertex_index["one"] = nv(graph)
# add_vertex!(graph, Dict(:count => 2))
# vertex_index["two"] = nv(graph)
# add_vertex!(graph, Dict(:count => 3))
# vertex_index["three"] = nv(graph)
# add_vertex!(graph, Dict(:count => 4))
# vertex_index["four"] = nv(graph)

# set_prop!(graph, vertex_index["four"], :count, 5)

# add_edge!(graph, 1, 2, Dict(:type => "normal", :count => 1))

# for node in vertices(graph)
#     print("$node ")
#     println(props(graph, node))
# end

# for node in edges(graph)
#     print("$node ")
#     println(props(graph, node))
# end