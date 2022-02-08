import os
import sys

from SourceCodeTools.code.data.file_utils import unpersist


def main():
    path = sys.argv[1]

    environments = sorted(list(filter(lambda path: os.path.isdir(path), (os.path.join(path, dir) for dir in os.listdir(path)))), key=lambda x: x.lower())
    for env_path in environments:

        edges_path = os.path.join(env_path, "edges_with_ast.bz2")
        if os.path.isfile(edges_path):
            edges = unpersist(edges_path)

            if any(edges["type"] == "prev_rev"):
                print()


if __name__ == "__main__":
    main()