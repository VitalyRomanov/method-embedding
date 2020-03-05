import pandas
import sys
from os.path import join, isdir
from os import mkdir

from data import load_data

nodes_path = sys.argv[1] #"/Volumes/External/datasets/Code/source-graphs/python-source-graph/02_largest_component/nodes_component_0.csv"
edges_path = sys.argv[2] #"/Volumes/External/datasets/Code/source-graphs/python-source-graph/02_largest_component/edges_component_0.csv"

nodes, edges = load_data(nodes_path, edges_path)



def split(edges):
    edges_shuffled = edges.sample(frac=1.)
    train_frac = int(edges_shuffled.shape[0] * 0.9)
    # print("Current train frac: ", train_frac)

    train = edges_shuffled\
        .iloc[:train_frac]
    test = edges_shuffled\
        .iloc[train_frac:]
    return train, test

def generate_edge_files(train, test, prefix):

    nodes_path = join(prefix,"nodes.csv")
    edges_test_path = join(prefix,"edges_test.csv")
    edges_train_path = join(prefix,"edges_train.csv")
    edges_train_512_1_path = join(prefix, "edges_512_1_train.csv")
    edges_train_512_1_16_2_path = join(prefix, "edges_512_1_16_2_train.csv")

    filtered_nodes = prefix.split("/")[-1]

    print("%s\t%s\t%s\t%s" % (filtered_nodes + "_all", nodes_path, edges_train_path, edges_test_path))
    print("%s\t%s\t%s\t%s" % (filtered_nodes + "_512_1", nodes_path, edges_train_512_1_path, edges_test_path))
    print("%s\t%s\t%s\t%s" % (filtered_nodes + "_512_1_16_2", nodes_path, edges_train_512_1_16_2_path, edges_test_path))

    # test.to_csv(edges_test_path, index=False)

    train.to_csv(edges_train_path, index=False)

    train.query("type != 512 and type != 1")\
        .to_csv(edges_train_512_1_path, index=False)

    train.query("type != 512 and type != 1 and type != 16 and type!= 2") \
        .to_csv(edges_train_512_1_16_2_path, index=False)

def filter_module(nodes, edges, module_names):

    # nodes = nodes.copy()

    nodes = nodes[
        nodes['name']\
            .apply(lambda x: x.split(".")[0])
            .apply(lambda x: x not in module_names)\
            .values
    ]

    ids = set(nodes['id'].values.tolist())

    edges = edges[
        edges['src'].apply(lambda x: x in ids).values
    ]
    edges = edges[
        edges['dst'].apply(lambda x: x in ids).values
    ]

    return nodes, edges



if not isdir("data"):
    mkdir("data")

def mkall(prefix, nodes, edges, excluded=[]):
    prefix = prefix + "_".join(excluded)
    if not isdir(prefix):
        mkdir(prefix)
    if excluded:
        nodes, edges = filter_module(nodes, edges, excluded)
    # train, test = split(edges)
    nodes.to_csv(join(prefix, "nodes.csv"), index=False)
    generate_edge_files(edges, edges, prefix)


mkall("data/full", nodes, edges)

mkall("data/no_", nodes, edges, excluded=['pandas'])
mkall("data/no_", nodes, edges, excluded=['pandas', 'ansible'])
mkall("data/no_", nodes, edges, excluded=['pandas', 'ansible', 'django'])
mkall("data/no_", nodes, edges, excluded=['pandas', 'ansible', 'django', 'numpy'])


# nodes, edges = filter_module(nodes, edges, ['pandas'])
# train, test = split(edges)
# nodes.to_csv("data/no_pandas/nodes.csv", index=False)
# generate_files(train, test, "data/no_pandas/edges")
#
# nodes, edges = filter_module(nodes, edges, ['pandas', 'ansible'])
# train, test = split(edges)
# nodes.to_csv("data/no_pandas_ansible/nodes.csv", index=False)
# generate_files(train, test, "data/no_pandas_ansible/")
#
# nodes, edges = filter_module(nodes, edges, ['pandas', 'ansible', 'django'])
# train, test = split(edges)
# nodes.to_csv("data/no_pandas_ansible_django/nodes.csv", index=False)
# generate_files(train, test, "data/no_pandas_ansible_django/edges")
#
# nodes, edges = filter_module(nodes, edges, ['pandas', 'ansible', 'django', 'numpy'])
# train, test = split(edges)
# nodes.to_csv("data/no_pandas_ansible_django_numpy/nodes.csv", index=False)
# generate_files(train, test, "data/no_pandas_ansible_django_numpy/edges")