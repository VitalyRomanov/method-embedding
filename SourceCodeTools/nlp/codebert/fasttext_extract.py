import pickle
from datetime import datetime
from pathlib import Path

import gensim
import numpy as np
from tqdm import tqdm

from SourceCodeTools.cli_arguments import TypePredictorTrainerArgumentParser
from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.nlp.entity.utils.data import read_data, read_json_data
from SourceCodeTools.nlp.trainers.cnn_entity_trainer import ModelTrainer, load_pkl_emb


def load_typed_nodes(path):
    from SourceCodeTools.code.data.file_utils import unpersist
    type_ann = unpersist(path)

    filter_rule = lambda name: "0x" not in name

    type_ann = type_ann[
        type_ann["dst"].apply(filter_rule)
    ]

    typed_nodes = set(type_ann["src"].tolist())
    return typed_nodes


class FasttextModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_type_ann_edges(self, path):
        self.type_ann_edges = path

    def get_batcher(self, *args, **kwargs):
        # kwargs.update({"tokenizer": "codebert"})
        return self.batcher(*args, **kwargs)

    def get_training_dir(self):
        if not hasattr(self, "_timestamp"):
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_params["model_output"]).joinpath("fasttext_extract_" + self._timestamp)

    def train_model(self):
        # graph_emb = load_pkl_emb(self.graph_emb_path) if self.graph_emb_path is not None else None
        word_emb = load_pkl_emb(self.word_emb_path)

        fasttext = gensim.models.FastText.load("/Users/LTV/Downloads/NitroShare/py_150_d500/model")

        typed_nodes = load_typed_nodes(self.type_ann_edges)
        vocab_mapping = word_emb.ind

        batcher = self.get_batcher(
            self.train_data + self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=None,
            wordmap=vocab_mapping, tagmap=None,
            class_weights=False, element_hash_size=1
        )

        node_ids = []
        embeddings = []

        added = set()

        for ind, batch in enumerate(tqdm(batcher)):
            for s_toks, s_repl in zip(batch["toks"], batch["replacements"]):
                unique_repls = set(list(s_repl))
                repls_for_ann = [r for r in unique_repls if r in typed_nodes]

                for r in repls_for_ann:
                    position = s_repl.index(r)
                    if position > 512:
                        continue
                    if r not in added:
                        node_ids.append(r)
                        embeddings.append(fasttext.wv[s_toks[position]])
                        added.add(r)

        all_embs = np.vstack(embeddings)
        embedder = Embedder(dict(zip(node_ids, range(len(node_ids)))), all_embs)

        output_path = self.get_training_dir()
        output_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(embedder, open(output_path.joinpath("fasttext_embeddings_500_2.pkl"), "wb"), fix_imports=False)
        print(node_ids)


def main():
    args = TypePredictorTrainerArgumentParser().parse()

    train_data, test_data = read_json_data(
        args.data_path, normalize=True, allowed=None, include_replacements=True,
        include_only="entities",
        min_entity_count=args.min_entity_count
    )

    trainer = FasttextModelTrainer(train_data, test_data, params={}, seq_len=512, word_emb_path=args.word_emb_path)
    trainer.set_type_ann_edges(args.type_ann_edges)
    trainer.train_model()



if __name__ == "__main__":
    main()