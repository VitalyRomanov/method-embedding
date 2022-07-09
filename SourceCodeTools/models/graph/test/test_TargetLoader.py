from tqdm import tqdm

from SourceCodeTools.code.common import read_edges


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("edges_file")
    parser.add_argument("positive_edges")
    parser.add_argument("negative_edges")

    args = parser.parse_args()

    existing_edges = set()

    for edges in tqdm(read_edges(args.edges_file, as_chunks=True), desc="Reading existing edges"):
        existing_edges.update(
            zip(edges["source_node_id"], edges["target_node_id"])
        )

    with open(args.positive_edges, "r") as positive:
        for line in tqdm(positive, desc="Testing positive"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                assert (int(parts[0]), int(parts[1])) in existing_edges

    # 1-2% of edges in negative appear in positive
    with open(args.negative_edges, "r") as negative:
        count = 0
        total = 0
        skipped = 0
        for line in tqdm(negative, desc="Testing negative"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                if (int(parts[0]), int(parts[1])) in existing_edges:
                    count += 1
                total += 1
            else:
                skipped += 1
                # assert (int(parts[0]), int(parts[1])) not in existing_edges

        print(f"Negative mistakes {count}/{total}, {count/total}, skipped {skipped}")

if __name__ == "__main__":
    main()