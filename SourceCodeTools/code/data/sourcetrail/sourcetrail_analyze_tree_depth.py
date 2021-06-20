import ast
import os
from typing import Iterable

from SourceCodeTools.code.data.sourcetrail.file_utils import unpersist
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.depth = 0

    def go(self, node, depth=0):
        depth += 1
        if depth > self.depth:
            self.depth = depth
        if isinstance(node, Iterable) and not isinstance(node, str):
            for subnode in node:
                self.go(subnode, depth=depth)
        if hasattr(node, "_fields"):
            for field in node._fields:
                self.go(getattr(node, field), depth=depth)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("bodies")
    args = parser.parse_args()

    bodies = unpersist(args.bodies)

    depths = []

    for ind, row in bodies.iterrows():
        body = row.body
        body_ast = ast.parse(body.strip())
        de = DepthEstimator()
        de.go(body_ast)
        depths.append(de.depth)

    print(f"Average depth: {sum(depths)/len(depths)}")
    depths = np.array(depths, dtype=np.int32)
    np.savetxt(os.path.join(os.path.dirname(args.bodies), "bodies_depths.txt"), depths, "%d")


if __name__ == "__main__":
    main()