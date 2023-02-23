import os
# from datetime import datetime
# from pathlib import Path

import torch
from SourceCodeTools.mltools.torch import get_length_mask
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaModel

from SourceCodeTools.nlp.trainers.cnn_entity_trainer import ModelTrainer


class CodeBertModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_gpu(self):
        if self.gpu_id != -1 and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.use_cuda = True
            self.device = f"cuda:{self.gpu_id}"
        else:
            self.use_cuda = False
            self.device = "cpu"

    def set_model_class(self):
        from SourceCodeTools.models.nlp.CodeBertSemiHybrid import CodeBertSemiHybridModel
        self.model = CodeBertSemiHybridModel

    def get_batcher(self, *args, **kwargs):
        kwargs.update({"tokenizer": "codebert"})
        return self.batcher(*args, **kwargs)

    def get_dataloaders(self, word_emb, graph_emb, suffix_prefix_buckets, **kwargs):
        decoder_mapping = RobertaTokenizer.from_pretrained("microsoft/codebert-base").decoder
        tok_ids, words = zip(*decoder_mapping.items())
        self._vocab_mapping = dict(zip(words, tok_ids))

        tagmap = kwargs.pop("tagmap", None)

        train_batcher = self.get_batcher(
            self.train_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=self.vocab_mapping, tagmap=tagmap,
            class_weights=False, element_hash_size=suffix_prefix_buckets, no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=self.vocab_mapping,
            tagmap=train_batcher.tagmap if tagmap is None else tagmap,  # use the same mapping
            class_weights=False, element_hash_size=suffix_prefix_buckets,  # class_weights are not used for testing
            no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        return train_batcher, test_batcher

    # def get_training_dir(self):
    #     if not hasattr(self, "_timestamp"):
    #         self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
    #     return Path(self.trainer_params["model_output"]).joinpath("codebert_" + self._timestamp)

    def get_model(self, *args, **kwargs):
        codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
        # codebert_model.config.attention_probs_dropout_prob
        model = self.model(
            codebert_model, graph_emb=kwargs["graph_embedder"],
            graph_padding_idx=kwargs["graph_padding_idx"],
            num_classes=kwargs["num_classes"],
            no_graph=self.no_graph
        )
        if self.use_cuda:
            model.cuda()

        if self.ckpt_path is not None:
            ckpt_path = os.path.join(self.ckpt_path, "checkpoint")
            model = self.load_checkpoint(model, ckpt_path)
        return model

    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

    def load_checkpoint(self, model, path):
        model.load_state_dict(torch.load(os.path.join(self.ckpt_path, "checkpoint"), map_location=torch.device('cpu')))
        return model

    def _load_word_embs(self):
        return None

    def _create_summary_writer(self, path):
        self.summary_writer = SummaryWriter(path)

    def _write_to_summary(self, label, value, step):
        self.summary_writer.add_scalar(label, value, step)

    @classmethod
    def _format_batch(cls, batch, device):
        key_types = {
            'tok_ids': torch.LongTensor,
            'tags': torch.LongTensor,
            'hide_mask': torch.BoolTensor,
            'no_loc_mask': torch.BoolTensor,
            'lens': torch.LongTensor,
            'graph_ids': torch.LongTensor,
            'graph_embs': torch.FloatTensor
        }
        for key, tf in key_types.items():
            if key in batch:
                batch[key] = tf(batch[key]).to(device)

    def _create_optimizer(self, model):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)

    def _lr_scheduler_step(self):
        self.scheduler.step()

    @classmethod
    def compute_loss_and_scores(
            cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph_embs=None,
            extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None,
            training=False
    ):
        token_ids[token_ids == len(vocab_mapping)] = vocab_mapping["<unk>"]
        seq_mask = get_length_mask(token_ids, lengths)
        logits = model(token_ids, graph_ids, graph_embs=graph_embs, mask=seq_mask, finetune=finetune)
        loss = model.loss(logits, labels, mask=seq_mask, class_weights=class_weights, extra_mask=extra_mask)
        scores = model.score(logits, labels, mask=seq_mask, scorer=scorer, extra_mask=extra_mask)

        scores["loss"] = loss

        return scores

    @classmethod
    def make_step(
            cls, model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, graph_embs=None,
            extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None, train=False
    ):

        token_ids[token_ids == 50265] = 1

        torch.set_grad_enabled(train)
        scores = cls.compute_loss_and_scores(
            model, token_ids, prefix, suffix, graph_ids, labels, lengths, graph_embs=graph_embs,
            extra_mask=extra_mask, class_weights=class_weights, scorer=scorer, finetune=finetune,
            vocab_mapping=vocab_mapping, training=train
        )
        torch.set_grad_enabled(True)

        if train is True:
            optimizer.zero_grad()
            scores["loss"].backward()
            optimizer.step()

        scores["loss"] = scores["loss"].cpu().item()

        return scores

    @staticmethod
    def set_model_training(model):
        model.train()

    @staticmethod
    def set_model_evaluation(model):
        model.eval()

    def train(
            self, model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
            learning_rate_decay=1., save_ckpt_fn=None, no_localization=False
    ):

        self._create_optimizer(model)

        train_scores, test_scores, train_average_scores, test_average_scores = self.iterate_epochs(
            train_batches, test_batches, epochs, model, scorer, save_ckpt_fn
        )

        return train_scores, test_scores, train_average_scores, test_average_scores
