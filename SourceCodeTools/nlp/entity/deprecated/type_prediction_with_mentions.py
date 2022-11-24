from __future__ import unicode_literals, print_function

import os
from SourceCodeTools.nlp.batchers import PythonBatcherMentions
from SourceCodeTools.nlp.entity.tf_models.params import att_params
from SourceCodeTools.nlp.entity.type_prediction import get_type_prediction_arguments, \
    ModelTrainer, save_entities
from SourceCodeTools.nlp.entity.utils import get_unique_entities
from SourceCodeTools.nlp.entity.utils.data import read_data


class ModelTrainerWithMentions(ModelTrainer):
    def __init__(self, train_data, test_data, params, graph_emb_path=None, word_emb_path=None,
            output_dir=None, epochs=30, batch_size=32, seq_len=100, finetune=False, trials=1):
        super(ModelTrainerWithMentions, self).__init__(
            train_data, test_data, params, graph_emb_path, word_emb_path,
            output_dir, epochs, batch_size, seq_len, finetune, trials
        )

    def set_batcher_class(self):
        self.batcher = PythonBatcherMentions

    def set_model_class(self):
        from SourceCodeTools.nlp.entity.tf_models.tf_model_with_mentions import TypePredictor
        self.model = TypePredictor

    def train(self, *args, **kwargs):
        from SourceCodeTools.nlp.entity.tf_models.tf_model_with_mentions import train
        return train(*args, **kwargs)


if __name__ == "__main__":
    args = get_type_prediction_arguments()

    output_dir = args.model_output
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # allowed = {'str', 'bool', 'Optional', 'None', 'int', 'Any', 'Union', 'List', 'Dict', 'Callable', 'ndarray',
    #            'FrameOrSeries', 'bytes', 'DataFrame', 'Matcher', 'float', 'Tuple', 'bool_t', 'Description', 'Type'}

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True, include_only="entities",
        min_entity_count=5
    )

    unique_entities = get_unique_entities(train_data, field="entities")
    save_entities(output_dir, unique_entities)

    for params in att_params:
        trainer = ModelTrainerWithMentions(
            train_data, test_data, params, graph_emb_path=args.graph_emb_path, word_emb_path=args.word_emb_path,
            output_dir=output_dir, epochs=args.epochs, batch_size=args.batch_size,
            finetune=args.finetune, trials=args.trials, seq_len=args.max_seq_len,
        )
        trainer.train_model()