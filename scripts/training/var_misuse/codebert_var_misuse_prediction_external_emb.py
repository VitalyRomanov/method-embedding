from pathlib import Path

from SourceCodeTools.code.data.cubert_python_benchmarks.data_iterators import DataIterator
from SourceCodeTools.nlp.trainers.codebert_entity_trainer_with_external_embeddings import \
    CodeBertModelTrainerWithExternalEmbeddings
from scripts.training.type_prediction.cnn_type_prediction import get_type_prediction_arguments


def main():
    args = get_type_prediction_arguments()

    data_path = Path(args.data_path)
    train_data = DataIterator(data_path, "train")
    test_data = DataIterator(data_path, "test")

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
