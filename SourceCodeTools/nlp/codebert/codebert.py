import pickle

import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel

from SourceCodeTools.models.Embedder import Embedder
from SourceCodeTools.nlp.entity.type_prediction import get_type_prediction_arguments, ModelTrainer
from SourceCodeTools.nlp.entity.utils.data import read_data


def load_typed_nodes(path):
    from SourceCodeTools.code.data.sourcetrail.file_utils import unpersist
    type_ann = unpersist(path)

    filter_rule = lambda name: "0x" not in name

    type_ann = type_ann[
        type_ann["dst"].apply(filter_rule)
    ]

    typed_nodes = set(type_ann["src"].tolist())
    return typed_nodes


class CodeBertModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_batcher(self, *args, **kwargs):
        kwargs.update({"tokenizer": "codebert"})
        return self.batcher(*args, **kwargs)

    def train_model(self):
        # graph_emb = load_pkl_emb(self.graph_emb_path) if self.graph_emb_path is not None else None

        typed_nodes = load_typed_nodes(self.args.type_ann_edges)

        decoder_mapping = RobertaTokenizer.from_pretrained("microsoft/codebert-base").decoder
        tok_ids, words = zip(*decoder_mapping.items())
        vocab_mapping = dict(zip(words, tok_ids))
        batcher = self.get_batcher(
            self.train_data + self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=None,
            wordmap=vocab_mapping, tagmap=None,
            class_weights=False, element_hash_size=1
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        model.to(device)

        node_ids = []
        embeddings = []

        for ind, batch in enumerate(tqdm(batcher)):
            # token_ids, graph_ids, labels, class_weights, lengths = b
            token_ids = torch.LongTensor(batch["tok_ids"])
            lens = torch.LongTensor(batch["lens"])

            token_ids[token_ids == len(vocab_mapping)] = vocab_mapping["<unk>"]

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
                    node_ids.append(r)
                    embeddings.append(s_emb[position])

        all_embs = torch.stack(embeddings, dim=0).numpy()
        embedder = Embedder(dict(zip(node_ids, range(len(node_ids)))), all_embs)
        pickle.dump(embedder, open("codebert_embeddings.pkl", "wb"), fix_imports=False)
        print(node_ids)


def main():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    # model.to(device)
    args = get_type_prediction_arguments()

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True,
        include_only="entities",
        min_entity_count=args.min_entity_count, random_seed=args.random_seed
    )

    trainer = CodeBertModelTrainer(train_data, test_data, params={}, seq_len=512)
    trainer.train_model()



if __name__ == "__main__":
    main()