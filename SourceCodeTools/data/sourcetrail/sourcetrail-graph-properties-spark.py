#%%
import os
os.environ["PYSPARK_DRIVER_PYTHON_OPTS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"
from pyspark import SparkContext
from pyspark.sql import SparkSession
import sys

sc = SparkContext.getOrCreate()
spark = SparkSession.builder.getOrCreate()

#%%

nodes_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/normalized_sourcetrail_nodes.csv"
edges_path = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/edges.csv"

#%%

nodes = spark.read.load(nodes_path, format="csv", header=True)
edges = spark.read.load(edges_path, format="csv", header=True)

# %%
edges = edges.withColumnRenamed("source_node_id", "src") \
            .withColumnRenamed("target_node_id", "dst") \
            .withColumnRenamed("id", "rel_id") \
            .withColumnRenamed("type", "rel_type")

# %%
from graphframes import *

g = GraphFrame(nodes, edges)


# %%
