from datetime import datetime
from pathlib import Path

from transformers import RobertaTokenizer

from SourceCodeTools.nlp.batchers.PythonBatcher import PythonBatcherWithGraphEmbeddings
from SourceCodeTools.nlp.trainers.codebert_entity_trainer import CodeBertModelTrainer


class CodeBertModelTrainerWithExternalEmbeddings(CodeBertModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_batcher_class(self):
        self.batcher = PythonBatcherWithGraphEmbeddings

    def _load_grap_embs(self):
        from SourceCodeTools.models.Embedder import EmbedderOnDisk
        return EmbedderOnDisk(self.graph_emb_path)

    def get_dataloaders(self, word_emb, graph_emb, suffix_prefix_buckets, **kwargs):
        decoder_mapping = RobertaTokenizer.from_pretrained("microsoft/codebert-base").decoder
        tok_ids, words = zip(*decoder_mapping.items())
        self._vocab_mapping = dict(zip(words, tok_ids))

        train_batcher = self.get_batcher(
            self.train_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb,
            wordmap=self.vocab_mapping, tagmap=None,
            class_weights=False, element_hash_size=suffix_prefix_buckets, no_localization=self.no_localization,
            mask_unlabeled_declarations=self.mask_unlabeled_declarations, **kwargs
        )
        test_batcher = self.get_batcher(
            self.test_data, self.batch_size, seq_len=self.seq_len,
            graphmap=graph_emb,
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
        return Path(self.trainer_params["model_output"]).joinpath("codebert_w_external_" + self._timestamp)
