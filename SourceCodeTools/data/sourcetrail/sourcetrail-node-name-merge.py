import sys, os
import pandas as p
from csv import QUOTE_NONNUMERIC
from SourceCodeTools.data.sourcetrail.sourcetrail_types import node_types
from SourceCodeTools.data.sourcetrail.file_utils import *

# needs testing
def normalize(line):
    line = line.replace('".	m', "")
    line = line.replace(".	m", "")
    line = line.replace("	m", "")
    line = line.replace("	s	p	n","#") # this will be replaced with .
    # the following are used by java
    # TODO
    # not all names processed correctly in java
    # .	morg	s	p	ndeeplearning4j	s	p	narbiter	s	p	nlayers	s	p	nLocalResponseNormalizationLayerSpace	s	p	nLocalResponseNormalizationLayerSpace	s	p(org.deeplearning4j.arbiter.layers.LocalResponseNormalizationLayerSpace.Builder)
    # LocalResponseNormalizationLayerSpace(org@deeplearning4j@arbiter@layers@LocalResponseNormalizationLayerSpace@Builder)
    #
    # no rule for this: s   p   nk
    # 1886698,2048,.	morg	s	p	ndeeplearning4j	s	p	narbiter	s	p	nlayers	s	p	nLocalResponseNormalizationLayerSpace	s	p	nk	sorg.deeplearning4j.arbiter.optimize.api.ParameterSpace<java.lang.Double>	p
    #
    # verify
    # 1886697,2048,"org.deeplearning4j.arbiter.layers.LocalResponseNormalizationLayerSpace.n___org@deeplearning4j@arbiter@optimize@api@ParameterSpace<java@lang@Double>___"
    # 1886698,2048,"org.deeplearning4j.arbiter.layers.LocalResponseNormalizationLayerSpace.k___org@deeplearning4j@arbiter@optimize@api@ParameterSpace<java@lang@Double>___"
    # 1886231,8192,"org.deeplearning4j.arbiter.layers.LayerSpace<L>.Builder<T>.dropOut___org@deeplearning4j@arbiter@layers@LayerSpace<L>@Builder<T>@T___(org@deeplearning4j@arbiter@optimize@api@ParameterSpace<java@lang@Double>)"
    # 1886232,8192,"org.deeplearning4j.arbiter.layers.LayerSpace<L>.Builder<T>.iDropOut___org@deeplearning4j@arbiter@layers@LayerSpace<L>@Builder<T>@T___(org@deeplearning4j@arbiter@optimize@api@ParameterSpace<org@deeplearning4j@nn@conf@dropout@IDropout>)"
    line = line.replace("	s	p(", "___(")
    line = line.replace('	s	p"', "")
    line = line.replace("	s	p", "")
    line = line.replace("	s", "___")
    line = line.replace("	p", "___")
    line = line.replace("	n", "___")
    line = line.replace('"',"")
    line = line.replace(" ", "_")
    line = line.replace(".", "@")
    # return the dot
    line = line.replace("#", ".")
    # if line.endswith("___"):
    #     line = line[:-3]
    return line

nodes_path = sys.argv[1]

# try:
data = unpersist_or_exit(nodes_path, exit_message="Sourcetrail nodes are empty", dtype={"id": int, "type": int, "serialized_name": str})
data = data[data['type'] != 262144] # filter nodes for files
data['serialized_name'] = data['serialized_name'].apply(normalize)
data['type'] = data['type'].apply(lambda x: node_types[x])

if len(data) > 0:
    data = data.astype({"id": int, "type": str, "serialized_name": str})
    persist(data, os.path.join(os.path.dirname(nodes_path), filenames["nodes"]))

# data.to_csv(os.path.join(os.path.dirname(nodes_path), "normalized_sourcetrail_nodes.csv"), index=False, quoting=QUOTE_NONNUMERIC)
# except p.errors.EmptyDataError:
#     with open(os.path.join(os.path.dirname(nodes_path), "normalized_sourcetrail_nodes.csv"), "w") as sink:
#         sink.write("id,type,serialized_name\n")
# except:
#     print("Error during merging")