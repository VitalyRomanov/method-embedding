from collections import OrderedDict
from itertools import chain

import datasets
import sacrebleu
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, NLLLoss

from SourceCodeTools.code.data.sourcetrail import SubwordMasker
from SourceCodeTools.models.graph.ElementEmbedder import DocstringEmbedder, create_fixed_length, \
    ElementEmbedderWithBpeSubwords
from SourceCodeTools.models.graph.train.objectives import SubwordEmbedderObjective
from SourceCodeTools.models.graph.train.objectives.AbstractObjective import AbstractObjective, _compute_accuracy
from SourceCodeTools.models.nlp.Decoder import LSTMDecoder
from SourceCodeTools.models.nlp.Vocabulary import Vocabulary
from SourceCodeTools.nlp.embed.bpe import load_bpe_model
import numpy as np


class GraphTextPrediction(SubwordEmbedderObjective):
    def __init__(
            self, graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=None, target_emb_size=None, link_predictor_type="inner_prod", masker: SubwordMasker = None,
            measure_ndcg=False, dilate_ndcg=1
    ):
        super().__init__(
            "GraphTextPrediction", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
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
            measure_ndcg=False, dilate_ndcg=1, max_len=3
    ):
        self.max_len = max_len
        super().__init__(
            "GraphTextGeneration", graph_model, node_embedder, nodes, data_loading_func, device,
            sampling_neighbourhood_size, batch_size,
            tokenizer_path=tokenizer_path, target_emb_size=target_emb_size, link_predictor_type=link_predictor_type,
            masker=masker, measure_ndcg=measure_ndcg, dilate_ndcg=dilate_ndcg
        )

    def create_target_embedder(self, data_loading_func, nodes, tokenizer_path):
        self.target_embedder = TextGenerationTargetMapper(
            elements=data_loading_func(), nodes=nodes, emb_size=self.target_emb_size,
            tokenizer_path=tokenizer_path, max_len=self.max_len
        ).to(self.device)

    def create_link_predictor(self):
        self.decoder = LSTMDecoder(
            self.target_embedder.num_buckets, padding=self.target_embedder.pad_id,
            encoder_embed_dim=self.target_emb_size, num_layers=2
        )

    def compute_logits(self, graph_emb, labels):
        prev_tokens = labels
        logits = self.decoder(prev_tokens, graph_emb)[:, :-1, :]
        return logits

    def compute_acc_loss(self, graph_emb, labels, lengths, return_logits=False):
        # prev_tokens = labels
        # logits = self.decoder(prev_tokens, graph_emb)[:, :-1, :]
        logits = self.compute_logits(graph_emb, labels)
        labels = labels[:, 1:]

        max_len = logits.shape[1]
        length_mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        # mask_ = length_mask.reshape(-1,)
        # loss_fct = NLLLoss(reduction="sum")
        loss = loss_fct(logits.reshape(-1, logits.size(-1)),#[mask_, :],
                        labels.reshape(-1))  #[mask_])

        def masked_accuracy(pred, true, mask):
            mask = mask.reshape(-1,)
            pred = pred.reshape(-1,)[mask]
            true = true.reshape(-1,)[mask]
            return _compute_accuracy(pred, true)

        acc = masked_accuracy(logits.argmax(dim=2), labels, length_mask)

        if return_logits:
            return acc, loss, logits
        return acc, loss


    def forward(self, input_nodes, seeds, blocks, train_embeddings=True):
        graph_emb = self._logits_batch(input_nodes, blocks, train_embeddings)
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
        total_loss = 0
        total_acc = 0
        total_bleu = {f"bleu": 0.}
        bleu_count = 0
        count = 0

        for input_nodes, seeds, blocks in getattr(self, f"{data_split}_loader"):
            blocks = [blk.to(self.device) for blk in blocks]

            src_embs = self._logits_batch(input_nodes, blocks)
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
            bleu_count += 1
            total_bleu["bleu"] += bleu['score']

            total_loss += loss.item()
            total_acc += acc
            count += 1
        return total_loss / count, total_acc / count, {"bleu": total_bleu["bleu"] / bleu_count}

    def evaluate(self, data_split, neg_sampling_factor=1):
        loss, acc, bleu = self.evaluate_generation(data_split)
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