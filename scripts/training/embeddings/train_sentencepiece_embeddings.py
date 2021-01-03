from SourceCodeTools.embed.bpe import load_bpe_model, make_tokenizer
from SourceCodeTools.embed.fasttext import train_fasttext
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('tokenizer_model_path', type=str, default=150, help='Path to sentencepiece tokenizer model')
parser.add_argument('input_file', type=str, default=150, help='Path to corpus')
parser.add_argument('output_dir', type=str, default=5, help='Output saving directory')
args = parser.parse_args()

train_fasttext(
    corpus_path=args.input_file,
    output_path=args.output_dir,
    tokenizer=make_tokenizer(load_bpe_model(args.tokenizer_model_path))
)

