#!/bin/bash
echo "type,count" > component_0_edge_type_count.csv
echo "in_degree,count" > component_0_in_degree_count.csv
echo "type,avg_in_degree" > component_0_node_avg_in_degree.csv
echo "type,avg_out_degree" > component_0_node_avg_out_degree.csv
echo "id,type,serialized_name,component,in_degree" > component_0_node_in_degrees.csv
echo "id,type,serialized_name,component,out_degree" > component_0_node_out_degrees.csv
echo "type,count" > component_0_node_type_count.csv
echo "out_degree,count" > component_0_out_degree_count.csv
echo "id,type,source_node_id,target_node_id" > edges_component_0.csv
echo "id,type,serialized_name,component" > nodes_component_0.csv


cat component_0_edge_type_count/* >> component_0_edge_type_count.csv
cat component_0_in_degree_count/* >> component_0_in_degree_count.csv
cat component_0_node_avg_in_degree/* >> component_0_node_avg_in_degree.csv
cat component_0_node_avg_out_degree/* >> component_0_node_avg_out_degree.csv
cat component_0_node_in_degrees/* >> component_0_node_in_degrees.csv
cat component_0_node_out_degrees/* >> component_0_node_out_degrees.csv
cat component_0_node_type_count/* >> component_0_node_type_count.csv
cat component_0_out_degree_count/* >> component_0_out_degree_count.csv
cat component_0_edges/* >> edges_component_0.csv
cat component_0_nodes/* >> nodes_component_0.csv

rm -rf component_0_edge_type_count
rm -rf component_0_in_degree_count
rm -rf component_0_node_avg_in_degree
rm -rf component_0_node_avg_out_degree
rm -rf component_0_node_in_degrees
rm -rf component_0_node_out_degrees
rm -rf component_0_node_type_count
rm -rf component_0_out_degree_count
rm -rf component_0_edges
rm -rf component_0_nodes

for file in *.csv; do
    bzip2 $file;
done

for file in *.bz2; do
  mv $file $(echo "$file" | awk -F"." '{print $1}').bz2;
done