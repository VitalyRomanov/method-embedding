from SourceCodeTools.cli_arguments import CodeBertEmbeddingExtractorArgumentParser, \
    default_graph_config
from SourceCodeTools.nlp.entity.utils.data import read_json_data
from SourceCodeTools.nlp.trainers.hybrid_entity_trainer import HybridModelTrainer


def main():
    config = CodeBertEmbeddingExtractorArgumentParser().parse()

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

    graph_config = default_graph_config()
    graph_config["TOKENIZER"]["tokenizer_path"] = "/Users/LTV/dev/method-embeddings/examples/sentencepiece_bpe.model"

    trainer = HybridModelTrainer(
        train_data, test_data,
        config=config,
        graph_config=graph_config,
    )
    trainer.train_model()


if __name__ == "__main__":
    main()
