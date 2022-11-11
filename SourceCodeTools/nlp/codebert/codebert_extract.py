import pickle
from datetime import datetime
from itertools import chain
from pathlib import Path

import torch
from SourceCodeTools.nlp.codebert.codebert_train import CodeBertModelTrainer
from tqdm import tqdm
from transformers import RobertaModel

from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.nlp.entity.type_prediction import get_type_prediction_arguments, ModelTrainer
from SourceCodeTools.nlp.entity.utils.data import read_data


def load_typed_nodes(path):
    from SourceCodeTools.code.data.file_utils import unpersist
    type_ann = unpersist(path)

    filter_rule = lambda name: "0x" not in name

    type_ann = type_ann[
        type_ann["dst"].apply(filter_rule)
    ]

    typed_nodes = set(type_ann["src"].tolist())
    return typed_nodes


class CodeBertModelExtractor(CodeBertModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_type_ann_edges(self, path):
        self.type_ann_edges = path

    def get_model(self, *args, **kwargs):
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        model.to(self.device)
        return model

    def get_training_dir(self):
        if not hasattr(self, "_timestamp"):
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_params["model_output"]).joinpath("codebert_extract_" + self._timestamp)

    def train_model(self):

        typed_nodes = load_typed_nodes(self.type_ann_edges)

        train_batcher, test_batcher = self.get_dataloaders(None, None, 1)

        model = self.get_model()

        node_ids = []
        embeddings = []

        added = set()

        for ind, batch in enumerate(tqdm(chain(train_batcher, test_batcher))):
            # token_ids, graph_ids, labels, class_weights, lengths = b
            token_ids = torch.LongTensor(batch["tok_ids"])
            lens = torch.LongTensor(batch["lens"])

            token_ids[token_ids == len(self.vocab_mapping)] = self.vocab_mapping["<unk>"]

            def get_length_mask(target, lens):
                mask = torch.arange(target.size(1)).to(target.device)[None, :] < lens[:, None]
                return mask

            mask = get_length_mask(token_ids, lens)
            with torch.no_grad():
                embs = model(input_ids=token_ids, attention_mask=mask)

            for s_emb, s_repl in zip(embs.last_hidden_state, batch["replacements"]):
                unique_repls = set(list(s_repl))
                repls_for_ann = [r for r in unique_repls if r in typed_nodes]

                for r in repls_for_ann:
                    position = s_repl.index(r)
                    if position > 512:
                        continue
                    if r not in added:
                        node_ids.append(r)
                        embeddings.append(s_emb[position])
                        added.add(r)

        all_embs = torch.stack(embeddings, dim=0).numpy()
        embedder = Embedder(dict(zip(node_ids, range(len(node_ids)))), all_embs)

        output_path = self.get_training_dir()
        output_path.mkdir(parents=True, exist_ok=True)
        pickle.dump(embedder, open(output_path.joinpath("codebert_embeddings.pkl"), "wb"), fix_imports=False)
        print(node_ids)


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    # model = RobertaModel.from_pretrained("microsoft/codebert-base")
    # model.to(device)
    args = get_type_prediction_arguments()

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True,
        include_only="entities",
        min_entity_count=args.min_entity_count, random_seed=args.random_seed
    )

    trainer_params = {
        "seq_len": 512
    }

    trainer = CodeBertModelExtractor(train_data, test_data, model_params={}, trainer_params=trainer_params)
    trainer.set_type_ann_edges(args.type_ann_edges)
    trainer.train_model()



if __name__ == "__main__":
    main()