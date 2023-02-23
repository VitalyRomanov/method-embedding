from SourceCodeTools.cli_arguments import CodeBERTTypePredictorTrainerArgumentParser
from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.codebert_entity_trainer import CodeBertModelTrainer


def main():
    config = CodeBERTTypePredictorTrainerArgumentParser().parse()

    if config["DATASET"]["restrict_allowed"]:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    train_data, test_data = read_json_data(
        config["DATASET"]["data_path"], normalize=True, allowed=allowed, include_replacements=True, include_only="entities",
        min_entity_count=config["DATASET"]["min_entity_count"]
    )

    config["TRAINING"]["suffix_prefix_buckets"] = 1

    trainer = CodeBertModelTrainer(
        train_data, test_data,
        config=config
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
