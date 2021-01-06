import sys


def main():
    sep = "\t"

    embs_path = sys.argv[1]
    tb_meta_path = sys.argv[2]
    tb_embs_path = sys.argv[3]

    with open(embs_path) as embeddings:
        embeddings.readline()
        with open(tb_meta_path, "w") as tb_meta:
            with open(tb_embs_path, "w") as tb_embs:
                for line in embeddings:
                    e_ = line.split(" ")
                    tb_meta.write(f"{e_[0]}\n")
                    tb_embs.write(f"{sep.join(e_[1:])}")


if __name__ == "__main__":
    main()
