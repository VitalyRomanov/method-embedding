import sys
from SourceCodeTools.nlp.embed.fasttext import export_w2v_for_tensorboard


def main():
    embs_path = sys.argv[1]
    tb_meta_path = sys.argv[2]
    tb_embs_path = sys.argv[3]

    export_w2v_for_tensorboard(embs_path, tb_meta_path, tb_embs_path)


if __name__ == "__main__":
    main()
