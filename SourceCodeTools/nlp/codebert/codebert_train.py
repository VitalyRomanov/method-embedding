import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from SourceCodeTools.mltools.torch import to_numpy
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaTokenizer, RobertaModel

from SourceCodeTools.nlp.entity.type_prediction import get_type_prediction_arguments, ModelTrainer, filter_labels

import torch.nn as nn


class CodebertHybridModel(nn.Module):
    def __init__(
            self, codebert_model, graph_emb, graph_padding_idx, num_classes, dense_hidden=100, dropout=0.1,
            bert_emb_size=768, no_graph=False
    ):
        super(CodebertHybridModel, self).__init__()

        self.codebert_model = codebert_model
        self.use_graph = not no_graph

        if self.use_graph:
            num_emb = graph_padding_idx + 1  # padding id is usually not a real embedding
            assert graph_emb is not None, "Graph embeddings are not provided, but model requires them"
            graph_emb_dim = graph_emb.e.shape[1]
            self.graph_emb = nn.Embedding(
                num_embeddings=num_emb, embedding_dim=graph_emb_dim, padding_idx=graph_padding_idx
            )

            pretrained_embeddings = torch.from_numpy(
                np.concatenate([graph_emb.e, np.zeros((1, graph_emb_dim))], axis=0)
            ).float()
            new_param = torch.nn.Parameter(pretrained_embeddings)
            assert self.graph_emb.weight.shape == new_param.shape
            self.graph_emb.weight = new_param
            self.graph_emb.weight.requires_grad = False
        else:
            graph_emb_dim = 0

        self.fc1 = nn.Linear(
            bert_emb_size + (graph_emb_dim if self.use_graph else 0),
            dense_hidden
        )
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_hidden, num_classes)

        self.loss_f = nn.CrossEntropyLoss(reduction="mean")

    def get_tensors_for_saliency(self, token_ids, graph_ids, mask):
        token_embs = self.codebert_model.embeddings.word_embeddings(token_ids)
        x = self.codebert_model(input_embeds=token_embs, attention_mask=mask).last_hidden_state

        if self.use_graph:
            graph_emb = self.graph_emb(graph_ids)
            x = torch.cat([x, graph_emb], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x)

        if self.use_graph:
            return token_embs, graph_emb, logits
        else:
            return token_embs, logits

    def forward(self, token_ids, graph_ids, mask, finetune=False):
        if finetune:
            x = self.codebert_model(input_ids=token_ids, attention_mask=mask).last_hidden_state
        else:
            with torch.no_grad():
                x = self.codebert_model(input_ids=token_ids, attention_mask=mask).last_hidden_state

        if self.use_graph:
            graph_emb = self.graph_emb(graph_ids)
            x = torch.cat([x, graph_emb], dim=-1)

        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x

    def loss(self, logits, labels, mask, class_weights=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        logits = logits[mask, :]
        labels = labels[mask]
        loss = self.loss_f(logits, labels)
        # if class_weights is None:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses, seq_mask))
        # else:
        #     loss = tf.reduce_mean(tf.boolean_mask(losses * class_weights, seq_mask))

        return loss

    def score(self, logits, labels, mask, scorer=None, extra_mask=None):
        if extra_mask is not None:
            mask = torch.logical_and(mask, extra_mask)
        true_labels = labels[mask]
        argmax = logits.argmax(-1)
        estimated_labels = argmax[mask]

        p, r, f1 = scorer(to_numpy(estimated_labels), to_numpy(true_labels))

        scores = {}
        scores["Precision"] = p
        scores["Recall"] = r
        scores["F1"] = f1

        return scores


def get_length_mask(target, lens):
    mask = torch.arange(target.size(1)).to(target.device)[None, :] < lens[:, None]
    return mask


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
        self.model = CodebertHybridModel

    def get_batcher(self, *args, **kwargs):
        kwargs.update({"tokenizer": "codebert"})
        return self.batcher(*args, **kwargs)

    def get_dataloaders(self, word_emb, graph_emb, suffix_prefix_buckets, **kwargs):
        decoder_mapping = RobertaTokenizer.from_pretrained("microsoft/codebert-base").decoder
        tok_ids, words = zip(*decoder_mapping.items())
        self._vocab_mapping = dict(zip(words, tok_ids))

        train_batcher = self.get_batcher(
            self.train_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=self.vocab_mapping, tagmap=None,
            class_weights=False, element_hash_size=suffix_prefix_buckets, no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb.ind if graph_emb is not None else None,
            wordmap=self.vocab_mapping,
            tagmap=train_batcher.tagmap,  # use the same mapping
            class_weights=False, element_hash_size=suffix_prefix_buckets,  # class_weights are not used for testing
            no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        return train_batcher, test_batcher

    def get_training_dir(self):
        if not hasattr(self, "_timestamp"):
            self._timestamp = str(datetime.now()).replace(":", "-").replace(" ", "_")
        return Path(self.trainer_params["model_output"]).joinpath("codebert_" + self._timestamp)

    def get_model(self, *args, **kwargs):
        codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
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
        model.load_state_dict(torch.load(os.path.join(self.ckpt_path, "checkpoint")))
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
            'graph_ids': torch.LongTensor
        }
        for key, tf in key_types.items():
            if key in batch:
                batch[key] = tf(batch[key]).to(device)

    def _create_optimizer(self, model):
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_decay)

    def _lr_scheduler_step(self):
        self.scheduler.step()

    # def train(
    #         self, model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
    #         learning_rate_decay=1., finetune=False, summary_writer=None, save_ckpt_fn=None, no_localization=False
    # ):
    #
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=learning_rate_decay)
    #
    #     train_losses = []
    #     test_losses = []
    #     train_f1s = []
    #     test_f1s = []
    #
    #     num_train_batches = len(train_batches)
    #     num_test_batches = len(test_batches)
    #
    #     best_f1 = 0.
    #
    #     try:
    #         for e in range(epochs):
    #             losses = []
    #             ps = []
    #             rs = []
    #             f1s = []
    #
    #             start = time()
    #             model.train()
    #
    #             for ind, batch in enumerate(tqdm(train_batches)):
    #                 self.format_batch(batch, self.device)
    #                 # token_ids, graph_ids, labels, class_weights, lengths = b
    #                 loss, p, r, f1 = self.make_step(
    #                     model=model, optimizer=optimizer, token_ids=batch['tok_ids'],
    #                     prefix=batch['prefix'], suffix=batch['suffix'], graph_ids=batch['graph_ids'],
    #                     labels=batch['tags'], lengths=batch['lens'],
    #                     extra_mask=batch['no_loc_mask'] if no_localization else batch['hide_mask'],
    #                     # class_weights=batch['class_weights'],
    #                     scorer=scorer, finetune=finetune and e / epochs > 0.6,
    #                     vocab_mapping=self.vocab_mapping,
    #                     train=True
    #                 )
    #                 losses.append(loss.cpu().item())
    #                 ps.append(p)
    #                 rs.append(r)
    #                 f1s.append(f1)
    #
    #                 self.summary_writer.add_scalar("Loss/Train", loss, global_step=e * num_train_batches + ind)
    #                 self.summary_writer.add_scalar("Precision/Train", p, global_step=e * num_train_batches + ind)
    #                 self.summary_writer.add_scalar("Recall/Train", r, global_step=e * num_train_batches + ind)
    #                 self.summary_writer.add_scalar("F1/Train", f1, global_step=e * num_train_batches + ind)
    #
    #             test_alosses = []
    #             test_aps = []
    #             test_ars = []
    #             test_af1s = []
    #
    #             model.eval()
    #
    #             for ind, batch in enumerate(test_batches):
    #                 self.format_batch(batch, self.device)
    #                 # token_ids, graph_ids, labels, class_weights, lengths = b
    #                 test_loss, test_p, test_r, test_f1 = self.make_step(
    #                     model=model, optimizer=None, token_ids=batch['tok_ids'],
    #                     prefix=batch['prefix'], suffix=batch['suffix'], graph_ids=batch['graph_ids'],
    #                     labels=batch['tags'], lengths=batch['lens'],
    #                     extra_mask=batch['no_loc_mask'] if no_localization else batch['hide_mask'],
    #                     # class_weights=batch['class_weights'],
    #                     scorer=scorer, vocab_mapping=self.vocab_mapping,
    #                     train=False
    #                 )
    #
    #                 self.summary_writer.add_scalar("Loss/Test", test_loss, global_step=e * num_test_batches + ind)
    #                 self.summary_writer.add_scalar("Precision/Test", test_p, global_step=e * num_test_batches + ind)
    #                 self.summary_writer.add_scalar("Recall/Test", test_r, global_step=e * num_test_batches + ind)
    #                 self.summary_writer.add_scalar("F1/Test", test_f1, global_step=e * num_test_batches + ind)
    #                 test_alosses.append(test_loss.cpu().item())
    #                 test_aps.append(test_p)
    #                 test_ars.append(test_r)
    #                 test_af1s.append(test_f1)
    #
    #             epoch_time = time() - start
    #
    #             train_losses.append(float(sum(losses) / len(losses)))
    #             train_f1s.append(float(sum(f1s) / len(f1s)))
    #             test_losses.append(float(sum(test_alosses) / len(test_alosses)))
    #             test_f1s.append(float(sum(test_af1s) / len(test_af1s)))
    #
    #             print(
    #                 f"Epoch: {e}, {epoch_time: .2f} s, Train Loss: {train_losses[-1]: .4f}, Train P: {sum(ps) / len(ps): .4f}, Train R: {sum(rs) / len(rs): .4f}, Train F1: {sum(f1s) / len(f1s): .4f}, "
    #                 f"Test loss: {test_losses[-1]: .4f}, Test P: {sum(test_aps) / len(test_aps): .4f}, Test R: {sum(test_ars) / len(test_ars): .4f}, Test F1: {test_f1s[-1]: .4f}")
    #
    #             if save_ckpt_fn is not None and float(test_f1s[-1]) > best_f1:
    #                 save_ckpt_fn()
    #                 best_f1 = float(test_f1s[-1])
    #
    #             scheduler.step(epoch=e)
    #
    #     except KeyboardInterrupt:
    #         pass
    #
    #     return train_losses, train_f1s, test_losses, test_f1s

    @classmethod
    def compute_loss_and_scores(
            cls, model, token_ids, prefix, suffix, graph_ids, labels, lengths,
            extra_mask=None, class_weights=None, scorer=None, finetune=False, vocab_mapping=None,
            training=False
    ):
        token_ids[token_ids == len(vocab_mapping)] = vocab_mapping["<unk>"]
        seq_mask = get_length_mask(token_ids, lengths)
        logits = model(token_ids, graph_ids, mask=seq_mask, finetune=finetune)
        loss = model.loss(logits, labels, mask=seq_mask, class_weights=class_weights, extra_mask=extra_mask)
        scores = model.score(logits, labels, mask=seq_mask, scorer=scorer, extra_mask=extra_mask)

        scores["loss"] = loss

        return scores

    @classmethod
    def make_step(
            cls, model, optimizer, token_ids, prefix, suffix, graph_ids, labels, lengths, extra_mask=None,
            class_weights=None, scorer=None, finetune=False, vocab_mapping=None, train=False
    ):

        if train is True:
            scores = cls.compute_loss_and_scores(
                model, token_ids, prefix, suffix, graph_ids, labels, lengths,
                extra_mask=extra_mask, class_weights=class_weights, scorer=scorer, finetune=finetune,
                vocab_mapping=vocab_mapping, training=True
            )
            optimizer.zero_grad()
            scores["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                scores = cls.compute_loss_and_scores(
                    model, token_ids, prefix, suffix, graph_ids, labels, lengths,
                    extra_mask=extra_mask, class_weights=class_weights, scorer=scorer, finetune=finetune,
                    vocab_mapping=vocab_mapping, training=False
                )

        scores["loss"] = scores["loss"].cpu().item()

        return scores

    def train(
            self, model, train_batches, test_batches, epochs, report_every=10, scorer=None, learning_rate=0.01,
            learning_rate_decay=1., finetune=False, save_ckpt_fn=None, no_localization=False
    ):

        self._create_optimizer(model)

        train_scores, test_scores, train_average_scores, test_average_scores = self.iterate_epochs(train_batches,
                                                                                                   test_batches, epochs,
                                                                                                   model, scorer,
                                                                                                   save_ckpt_fn)

        return train_scores, test_scores, train_average_scores, test_average_scores

    # def train_model(self):
    #
    #     print(f"\n\n{self.model_params}")
    #     lr = self.model_params.pop("learning_rate")
    #     lr_decay = self.model_params.pop("learning_rate_decay")
    #     suffix_prefix_buckets = self.model_params.pop("suffix_prefix_buckets")
    #
    #     graph_emb = load_pkl_emb(self.graph_emb_path) if self.graph_emb_path is not None else None
    #
    #     train_batcher, test_batcher = self.get_dataloaders(None, graph_emb, suffix_prefix_buckets=suffix_prefix_buckets)
    #
    #     codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base")
    #     model = CodebertHybridModel(
    #         codebert_model, graph_emb.e if graph_emb is not None else None, padding_idx=train_batcher.graphpad, num_classes=train_batcher.num_classes,
    #         no_graph=self.no_graph
    #     )
    #     if self.use_cuda:
    #         model.cuda()
    #
    #     trial_dir = self.get_trial_dir()
    #     os.makedirs(trial_dir)
    #     self.create_summary_writer(trial_dir)
    #
    #     def save_ckpt_fn():
    #         checkpoint_path = os.path.join(trial_dir, "checkpoint")
    #         torch.save(model, open(checkpoint_path, 'wb'))
    #
    #     train_losses, train_f1, test_losses, test_f1 = self.train(
    #         model=model, train_batches=train_batcher, test_batches=test_batcher,
    #         epochs=self.epochs, learning_rate=lr,
    #         scorer=lambda pred, true: scorer(pred, true, train_batcher.tagmap, no_localization=self.no_localization),
    #         learning_rate_decay=lr_decay, finetune=self.finetune, save_ckpt_fn=save_ckpt_fn,
    #         no_localization=self.no_localization
    #     )
    #
    #     # checkpoint_path = os.path.join(trial_dir, "checkpoint")
    #     # model.save_weights(checkpoint_path)
    #
    #     metadata = {
    #         "train_losses": train_losses,
    #         "train_f1": train_f1,
    #         "test_losses": test_losses,
    #         "test_f1": test_f1,
    #         "learning_rate": lr,
    #         "learning_rate_decay": lr_decay,
    #         "epochs": self.epochs,
    #         "suffix_prefix_buckets": suffix_prefix_buckets,
    #         "seq_len": self.seq_len,
    #         "batch_size": self.batch_size,
    #         "no_localization": self.no_localization
    #     }
    #
    #     print("Maximum f1:", max(test_f1))
    #
    #     # write_config(trial_dir, params, extra_params={"suffix_prefix_buckets": suffix_prefix_buckets, "seq_len": seq_len})
    #
    #     metadata.update(self.model_params)
    #
    #     with open(os.path.join(trial_dir, "params.json"), "w") as metadata_sink:
    #         metadata_sink.write(json.dumps(metadata, indent=4))
    #
    #     pickle.dump(train_batcher.tagmap, open(os.path.join(trial_dir, "tag_types.pkl"), "wb"))
    #
    #     # for ind, batch in enumerate(tqdm(batcher)):
    #     #     # token_ids, graph_ids, labels, class_weights, lengths = b
    #     #     token_ids = torch.LongTensor(batch["tok_ids"])
    #     #     lens = torch.LongTensor(batch["lens"])
    #     #
    #     #     token_ids[token_ids == len(vocab_mapping)] = vocab_mapping["<unk>"]
    #     #
    #     #     def get_length_mask(target, lens):
    #     #         mask = torch.arange(target.size(1)).to(target.device)[None, :] < lens[:, None]
    #     #         return mask
    #     #
    #     #     mask = get_length_mask(token_ids, lens)
    #     #     with torch.no_grad():
    #     #         embs = model(input_ids=token_ids, attention_mask=mask)
    #     #
    #     #     for s_emb, s_repl in zip(embs.last_hidden_state, batch["replacements"]):
    #     #         unique_repls = set(list(s_repl))
    #     #         repls_for_ann = [r for r in unique_repls if r in typed_nodes]
    #     #
    #     #         for r in repls_for_ann:
    #     #             position = s_repl.index(r)
    #     #             if position > 512:
    #     #                 continue
    #     #             node_ids.append(r)
    #     #             embeddings.append(s_emb[position])
    #     #
    #     # all_embs = torch.stack(embeddings, dim=0).numpy()
    #     # embedder = Embedder(dict(zip(node_ids, range(len(node_ids)))), all_embs)
    #     # pickle.dump(embedder, open("codebert_embeddings.pkl", "wb"), fix_imports=False)
    #     # print(node_ids)


def main():
    args = get_type_prediction_arguments()

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}
    if args.restrict_allowed:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    # train_data, test_data = read_data(
    #     open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True,
    #     include_only="entities",
    #     min_entity_count=args.min_entity_count, random_seed=args.random_seed
    # )

    from pathlib import Path
    dataset_dir = Path(args.data_path)
    train_data = filter_labels(
        pickle.load(open(dataset_dir.joinpath("type_prediction_dataset_no_defaults_train.pkl"), "rb")),
        allowed=allowed
    )
    test_data = filter_labels(
        pickle.load(open(dataset_dir.joinpath("type_prediction_dataset_no_defaults_test.pkl"), "rb")),
        allowed=allowed
    )

    trainer_params = args.__dict__
    trainer_params["suffix_prefix_buckets"] = 1

    trainer = CodeBertModelTrainer(
        train_data, test_data,
        model_params={},
        trainer_params=trainer_params
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
