from collections import OrderedDict, defaultdict
from itertools import chain

import datasets
import torch
from torch.nn import CrossEntropyLoss

from SourceCodeTools.code.data.dataset import SubwordMasker
from SourceCodeTools.models.graph.ElementEmbedder import DocstringEmbedder, create_fixed_length, \
    ElementEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.train.objectives import SubwordEmbedderObjective
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import compute_accuracy, \
    sum_scores
from SourceCodeTools.models.nlp.TorchDecoder import Decoder
from SourceCodeTools.models.nlp.Vocabulary import Vocabulary
import numpy as np


class GraphTextPrediction(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1
    ):
        super().__init__(
            "GraphTextPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = DocstringEmbedder(
            elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size,
            tokenizer_path=tokenizer_path
        ).to(self.device)


class GraphTextGeneration(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_scores=False, dilate_scores=1, max_len=20
    ):
        self.max_len = max_len + 2  # add pad and eos
        super().__init__(
            "GraphTextGeneration", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_scores=measure_scores, dilate_scores=dilate_scores
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = TextGenerationTargetMapper(
            elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size,
            tokenizer_path=tokenizer_path, max_len=self.max_len
        ).to(self.device)

    def create_link_predictor(self):
        # self.decoder = LSTMDecoder(
        #     self.target_embedder.num_buckets, padding=self.target_embedder.pad_id,
        #     encoder_embed_dim=self.target_emb_size, num_layers=2
        # ).to(self.device)

        self.decoder = Decoder(
            self.target_emb_size, decoder_dim=100, out_dim=self.target_embedder.num_buckets,
            vocab_size=self.target_embedder.num_buckets, nheads=4, layers=4
        ).to(self.device)

    def compute_logits(self, graph_emb, labels):
        prev_tokens = labels
        # logits = self.decoder(prev_tokens, graph_emb)[:, :-1, :]
        logits = self.decoder(graph_emb.unsqueeze(1), prev_tokens)[:, :-1, :]
        return logits

    def compute_acc_loss(self, graph_emb, labels, lengths, return_logits=False):
        # prev_tokens = labels
        # logits = self.decoder(prev_tokens, graph_emb)[:, :-1, :]
        logits = self.compute_logits(graph_emb, labels)
        labels = labels[:, 1:]

        max_len = logits.shape[1]
        length_mask = torch.arange(max_len).to(self.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        logits_unrolled = logits[length_mask, :]
        labels_unrolled = labels[length_mask]

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(logits_unrolled, labels_unrolled)
        # mask_ = length_mask.reshape(-1,)
        # loss_fct = NLLLoss(reduction="sum")
        # loss = loss_fct(logits.reshape(-1, logits.size(-1)),#[mask_, :],
        #                 labels.reshape(-1))  #[mask_])

        def masked_accuracy(pred, true, mask):
            mask = mask.reshape(-1,)
            pred = pred.reshape(-1,)[mask]
            true = true.reshape(-1,)[mask]
            return compute_accuracy(pred, true)

        acc = masked_accuracy(logits.argmax(dim=2), labels, length_mask)

        if return_logits:
            return acc, loss, logits
        return acc, loss

    def forward(self, input_nodes, seeds, blocks, train_embeddings=True, neg_sampling_strategy=None):
        graph_emb = self._graph_embeddings(input_nodes, blocks, train_embeddings)
        indices = self.seeds_to_global(seeds).tolist()
        labels, lengths = self.target_embedder[indices]
        labels = labels.to(self.device)
        lengths = lengths.to(self.device)
        acc, loss = self.compute_acc_loss(graph_emb, labels, lengths)

        return loss, acc

    def get_generated(self, tokens):
        eos_id = self.target_embedder.vocab.eos_id
        sents = []
        for sent in tokens.tolist():
            s = []
            for t in sent:
                if t != eos_id:
                    s.append(self.target_embedder.vocab[t])
                else:
                    break
            sents.append("".join(s).replace("▁"," "))

        return sents

    def evaluate_generation(self, data_split):
        scores = defaultdict(list)
        count = 0

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._graph_embeddings(input_nodes, blocks)
            indices = self.seeds_to_global(seeds).tolist()
            labels, lengths = self.target_embedder[indices]
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            acc, loss, logits = self.compute_acc_loss(src_embs, labels, lengths, return_logits=True)

            true = self.get_generated(labels[:, 1:])  # first token is <pad>
            pred = self.get_generated(logits.argmax(2))

            if not hasattr(self, "bleu_metric"):
                self.bleu_metric = datasets.load_metric('sacrebleu')

            # bleu = sacrebleu.corpus_bleu(pred, true)

            bleu = self.bleu_metric.compute(predictions=pred, references=[[t] for t in true])
            scores["bleu"].append(bleu['score'])

            scores["Loss"].append(loss.item())
            scores["Accuracy"].append(acc)
            count += 1

        scores = {key: sum_scores(val) for key, val in scores.items()}
        return scores

    def evaluate(self, data_split, *, neg_sampling_strategy=None, early_stopping=False, early_stopping_tolerance=20):
        loss, acc, bleu = self.evaluate_generation(data_split)
        if data_split == "val":
            self.check_early_stopping(acc)
        return loss, acc, bleu

    def parameters(self, recurse: bool = True):
        return chain(self.target_embedder.parameters())

    def custom_state_dict(self):
        state_dict = OrderedDict()
        for k, v in self.target_embedder.state_dict().items():
            state_dict[f"target_embedder.{k}"] = v
        return state_dict


class TextGenerationTargetMapper(ElementEmbedderWithBpeSubwords):
    def __init__(self, elements, nodes, emb_size, tokenizer_path, num_buckets=100000, max_len=100):
        super().__init__(
            elements=elements, nodes=nodes, emb_size=emb_size, num_buckets=num_buckets,
            max_len=max_len, tokenizer_path=tokenizer_path
        )

    def init_subwords(self, elements, num_buckets, max_len):
        from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
        self.tokenizer = load_bpe_model(self.tokenizer_path)
        tokenize = make_tokenizer(self.tokenizer)
        self.vocab = Vocabulary()
        self.num_buckets = self.tokenizer.vocab_size()
        self.pad_id = self.vocab.pad_id

        docs = self.elements['dst']
        tokens = docs.map(lambda text: ["<pad>"] + tokenize(text) + ["</s>"])
        lengths = tokens.map(lambda tokens: min(len(tokens) - 1, max_len))  # the pad will go away
        [self.vocab.add(doc) for doc in tokens]
        self.num_buckets = len(self.vocab)

        reprs = tokens \
            .map(lambda tokens: map(lambda token: self.vocab[token], tokens))\
            .map(lambda int_tokens: np.fromiter(int_tokens, dtype=np.int32))\
            .map(lambda parts: create_fixed_length(parts, max_len, self.pad_id))

        self.id2repr = dict(zip(self.elements["id"], reprs))
        self.id2len = dict(zip(self.elements["id"], lengths))

    def __getitem__(self, ids):
        """
        Get possible targets
        :param ids: Takes a list of original ids
        :return: Matrix with subwords for passing to embedder and the list of lengths
        """
        tokens = np.array([self.id2repr[id] for id in ids], dtype=np.int32)
        lengths = np.array([self.id2len[id] for id in ids], dtype=np.int32)
        return torch.LongTensor(tokens), torch.LongTensor(lengths)

    def set_embed(self):
        pass