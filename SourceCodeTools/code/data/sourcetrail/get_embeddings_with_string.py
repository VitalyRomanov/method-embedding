import pickle
import sys

import pandas as pd

from SourceCodeTools.code.data.file_utils import unpersist

# get tb embedding where labels are human-readable strings


def main():
    nodes_with_strings = sys.argv[1]
    embeddings = sys.argv[2]
    out_name_for_tb = sys.argv[3]

    all_embs = pickle.load(open(embeddings, "rb"))[0]
    nodes = unpersist(nodes_with_strings)

    embs = []
    strings = []

    tb_meta_path = out_name_for_tb + "meta.tsv"
    tb_embs_path = out_name_for_tb + "embs.tsv"

    nodes.dropna(inplace=True)
    nodes = nodes.sample(n=5000)

    with open(tb_meta_path, "w") as tb_meta:
        with open(tb_embs_path, "w") as tb_embs:
            tb_meta.write("string\ttype\n")
            for id, string, type in zip(nodes["id"], nodes["string"], nodes["type"]):
                if string is not None and not pd.isna(string) and "srctrl" not in string and "\n" not in string:
                    # embs.append(
                    #     all_embs[id]
                    # )
                    # strings.append(string)
                    sep = "\t"

                    try:
                        string =  f"{string}\t{type}\n"  #f"{string.encode('utf-8')}\n"
                        emb_string = f"{sep.join(str(e) for e in all_embs[int(id)])}\n"
                        tb_meta.write(string)
                        tb_embs.write(emb_string)
                    except:
                        pass




if __name__ == "__main__":
    main()