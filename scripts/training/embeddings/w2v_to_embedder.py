import pickle
from pathlib import Path

from SourceCodeTools.nlp.embed.fasttext import load_w2v_map


def w2v_to_embedder(w2v_path, embedder_path):
    w2v_path = Path(w2v_path)
    embedder_path = Path(embedder_path)

    embedder = load_w2v_map(w2v_path)
    pickle.dump(embedder, open(embedder_path, "wb"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("w2v_path")
    parser.add_argument("output_path")

    args = parser.parse_args()
    w2v_to_embedder(args.w2v_path, args.output_path)
