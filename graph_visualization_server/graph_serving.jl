using LightGraphs, MetaGraphs
using Dates
using JSON
using JLD
using HTTP

include("build_graph.jl")

struct NamedGraph
    mgraph
    index
    inv_index
end

# GRAPH = "graph.jld"

# @load GRAPH graph
# graph = JLD.load(GRAPH, "graph")
graph = build_graph()
println("finished building")


function all_neighbors(g, v)
    return Set(vcat(outneighbors(g, v), inneighbors(g, v)))
end

function use_edge(g, e)
    link_type = get_prop(g, e, :type)
    link_count = get_prop(g, e, :count)
    if link_type != "pattern_neighbour" && link_count > 0 #|| link_type == "pattern_neighbour" && link_count > 0
        return true
    end
    return false
end

function use_node(g, v)
    count = get_prop(g, v, :count)
    if count > 10
        return true
    else
        return false
    end
end

function get_valid_nodes_and_links(g, node_id, neigh, direction)
    links = []
    connected = copy(neigh)
    for ov in neigh
        if direction == "f"
            e = Edge(node_id, ov)
        else
            e = Edge(ov, node_id)
        end

        if has_edge(g.mgraph, e)
            if use_edge(g.mgraph, e)
                push!(links, e)
                continue
            end
        end
        pop!(connected, ov)
    end
    return connected, links
end

function expand_links(g, nodes)
    extra_links = Set()
    for v1 in nodes, v2 in nodes
        e = Edge(v1,v2)
        if has_edge(g.mgraph, e)
            if use_edge(g.mgraph, e)
                push!(extra_links, e)
                continue
            end
        end
    end
    return extra_links
end


function get_egonet(g, node_name)
    links = Set()
    nodes = Set()
    if node_name in keys(g.index)
        node_id = g.index[node_name]

        push!(nodes, node_id)

        out_nodes = Set(filter(v -> use_node(g.mgraph, v), outneighbors(g.mgraph, node_id)))
        c_n, c_l = get_valid_nodes_and_links(g, node_id, out_nodes, "f")
        extra_links = expand_links(g, c_n)
        nodes = union(nodes, c_n)
        links = union(links, c_l)
        links = union(links, extra_links)
        

        in_nodes = Set(filter(v -> use_node(g.mgraph, v), inneighbors(g.mgraph, node_id)))
        c_n, c_l = get_valid_nodes_and_links(g, node_id, in_nodes, "r")
        extra_links = expand_links(g, c_n)
        nodes = union(nodes, c_n)
        links = union(links, c_l)
        links = union(links, extra_links)

        node_set = nodes
        link_set = links


        # ego = Set(vcat(neighborhood(g.mgraph, node_id, 1, dir=:in), neighborhood(g.mgraph, node_id, 1, dir=:out)))
        # ego = Set(neighborhood(g.mgraph, node_id, 1, dir=:in))
        # ego = all_neighbors(g.mgraph, node_id)
        if length(node_set) > 0
            return node_set, link_set
        end
    end
    return 
end

function into_json(g, ego)
    # if length(ego) > 0
    #     links = []
    #     # for v in ego, n in all_neighbors(g, v)
    #     filtered_nodes = copy(ego)
    #     for v in ego
    #         e = Edge(v, node_id)
    #         if has_edge(g.mgraph, e)
    #             link_type = get_prop(g.mgraph, e, :type)
    #             link_count = get_prop(g.mgraph, e, :count)
    #             if link_type == "pattern_neighbour" || link_count < 5
    #                 pop!(filtered_nodes, v)
    #                 continue
    #             end
    #             push!(links, Dict("source" => g.inv_index[e.src], "target" => g.inv_index[e.dst], "type" => get_prop(g.mgraph, e, :type), "count" => get_prop(g.mgraph, e, :count)))
    #         end
    #     end
    # end

    n, l = ego

    nodes = [Dict("name" => g.inv_index[v]) for v in n]
    links = [Dict("source" => g.inv_index[e.src], "target" => g.inv_index[e.dst], "type" => get_prop(g.mgraph, e, :type), "count" => get_prop(g.mgraph, e, :count)) for e in l]
    message = JSON.json(Dict("nodes" => nodes, "links" => links))
    return message
end

function answer_request(g, node_name)
    ego = get_egonet(g, node_name)
    if ego === nothing
        return ""
    else
        return into_json(g, ego)
    end
end

# println(answer_request(graph, "banana"))

HTTP.listen("127.0.0.1", 8081) do http
    request_str = http.message.target
    request_str = replace(request_str[2:end], "%20" => " ")
    println(request_str)
    # @show http.message
    # @show HTTP.header(http, "Content-Type")
    # while !eof(http)
    #     println("body data: ", String(readavailable(http)))
    # end
    HTTP.setstatus(http, 200)
    HTTP.setheader(http, "Content-Type" => "application/json")
    HTTP.setheader(http, "Access-Control-Allow-Origin" => "*")
    startwrite(http)
    write(http, answer_request(graph, request_str))
    # write(http, "response body")
    # write(http, "more response body")
end