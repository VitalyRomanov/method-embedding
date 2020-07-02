import sys, os
import pandas as p
from csv import QUOTE_NONNUMERIC

# needs testing
def normalize(line):
    line = line.replace('".	m', "")
    line = line.replace(".	m", "")
    line = line.replace("	s	p	n","#") # this will be replaced with .
    # the following are used by java
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
    return line

nodes_path = sys.argv[1]

# try:
data = p.read_csv(nodes_path)
data = data[data['type'] != 262144]
data['serialized_name'] = data['serialized_name'].apply(normalize)

data.to_csv(os.path.join(os.path.dirname(nodes_path), "normalized_sourcetrail_nodes.csv"), index=False, quoting=QUOTE_NONNUMERIC)
# except p.errors.EmptyDataError:
#     with open(os.path.join(os.path.dirname(nodes_path), "normalized_sourcetrail_nodes.csv"), "w") as sink:
#         sink.write("id,type,serialized_name\n")
# except:
#     print("Error during merging")