from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.codebert_entity_trainer_with_external_embeddings import \
    CodeBertModelTrainerWithExternalEmbeddings
from scripts.training.type_prediction.cnn_type_prediction import get_type_prediction_arguments


def main():
    args = get_type_prediction_arguments()

    if args.restrict_allowed:
        allowed = {
            'str', 'Optional', 'int', 'Any', 'Union', 'bool', 'Callable', 'Dict', 'bytes', 'float', 'Description',
            'List', 'Sequence', 'Namespace', 'T', 'Type', 'object', 'HTTPServerRequest', 'Future', "Matcher"
        }
    else:
        allowed = None

    train_data, test_data = read_json_data(
        args.data_path, normalize=True, allowed=allowed, include_replacements=True, include_only="entities",
        min_entity_count=args.min_entity_count, random_seed=args.random_seed
    )

    trainer_params = args.__dict__
    trainer_params["suffix_prefix_buckets"] = 1

    trainer = CodeBertModelTrainerWithExternalEmbeddings(
        train_data, test_data,
        model_params={},
        trainer_params=trainer_params
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
