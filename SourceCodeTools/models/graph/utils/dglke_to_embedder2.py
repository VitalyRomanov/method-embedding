import pickle

import numpy
import pandas as pd

from SourceCodeTools.models.Embedder import Embedder


def read_entities(path):
    return pd.read_csv(path, sep="\t", header=None)[1]


def read_vectors(path):
    return numpy.load(path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("entities")
    parser.add_argument("vectors")
    parser.add_argument("output")

    args = parser.parse_args()

    entities = read_entities(args.entities)
    vectors = read_vectors(args.vectors)

    embedder_dict = dict()
    for entity, e_id in zip(entities, range(len(entities))):
        try:
            entity = int(entity)
        except:
            pass

        embedder_dict[entity] = e_id

    embedder = Embedder(embedder_dict, vectors)
    with open(args.output, "wb") as sink:
        pickle.dump(embedder, sink)





if __name__=="__main__":
    main()