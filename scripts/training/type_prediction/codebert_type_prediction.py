from SourceCodeTools.cli_arguments import CodeBERTTypePredictorTrainerArgumentParser
from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.codebert_entity_trainer import CodeBertModelTrainer


def main():
    args = CodeBERTTypePredictorTrainerArgumentParser().parse()

    if args.restrict_allowed:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    train_data, test_data = read_json_data(
        args.data_path, normalize=True, allowed=allowed, include_replacements=True, include_only="entities",
        min_entity_count=args.min_entity_count
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
