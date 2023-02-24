from pathlib import Path

root_path = Path(__file__).parent.parent


def test_cat_sources():
    input_dir = root_path.joinpath("res", "python_testdata", "example_code")
    output_file = root_path.joinpath("res", "python_testdata", "fasttext_pretr", "sources.txt")

    from scripts.training.embeddings.python_dir_cat import cat_files
    cat_files(input_dir, output_file)


def test_embedding_pretraining():
    input_file = root_path.joinpath("res", "python_testdata", "fasttext_pretr", "sources.txt")
    output_dir = root_path.joinpath("res", "python_testdata", "fasttext_pretr", "fasttext_100")

    from SourceCodeTools.nlp.embed.fasttext import train_fasttext
    from SourceCodeTools.nlp import create_tokenizer
    train_fasttext(
        corpus_path=input_file,
        output_path=output_dir,
        tokenizer=create_tokenizer("regex"),
        emb_size=100
    )


def test_w2v_to_embedder():
    from scripts.training.embeddings.w2v_to_embedder import w2v_to_embedder
    input_file = root_path.joinpath("res", "python_testdata", "fasttext_pretr", "fasttext_100", "emb.txt")
    output_file = root_path.joinpath("res", "python_testdata", "fasttext_pretr", "fasttext_100", "embedder.pkl")

    w2v_to_embedder(input_file, output_file)
