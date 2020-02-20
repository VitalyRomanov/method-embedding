import sys
import pandas as p

# needs testing
def normalize(line):
    # print(line)
    
    line = line.replace('".	m', "")
    line = line.replace(".	m", "")
    line = line.replace("	s	p	n",".")
    line = line.replace('	s	p"', "")
    line = line.replace('"',"")
    try:
        end_position = line.index("	")
    except:
        end_position = -1
    return line[:end_position]
    # parts = line.split("\t")
    # # print(parts)
    # parts = list(filter(lambda x: x not in {'.', 's', 'p', ''}, parts))
    # # print(parts)
    # parts = list(map(lambda x: x[1:] if len(x)>1 else Exception("Too short %s, %s" % (line, x)), parts))
    # # print(parts)
    # as_line = ".".join(parts)
    # return as_line

data = p.read_csv(sys.argv[1])
data = data[data['type'] != 262144]
data['serialized_name'] = data['serialized_name'].apply(normalize)

data.to_csv("normalized_sourcetrail_nodes.csv", index=False)