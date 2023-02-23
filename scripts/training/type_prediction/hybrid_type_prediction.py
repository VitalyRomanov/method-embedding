from pathlib import Path

from SourceCodeTools.cli_arguments import TypePredictorTrainerArgumentParser
from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.hybrid_entity_trainer import HybridModelTrainer


def main():
    args = TypePredictorTrainerArgumentParser().parse()

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

    trainer = HybridModelTrainer(
        train_data, test_data,
        model_params={},
        trainer_params=trainer_params,
        graph_subword_tokenizer_path=Path(__file__).parent.parent.parent.parent.joinpath("examples", "sentencepiece_bpe.model")
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
