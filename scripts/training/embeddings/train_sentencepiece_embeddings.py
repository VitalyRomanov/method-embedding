from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
from SourceCodeTools.nlp.embed.fasttext import train_wor2vec
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('tokenizer_model_path', type=str, default=150, help='Path to sentencepiece tokenizer model')
parser.add_argument('input_file', type=str, default=150, help='Path to corpus')
parser.add_argument('output_dir', type=str, default=5, help='Output saving directory')
args = parser.parse_args()

train_wor2vec(
    corpus_path=args.input_file,
    output_path=args.output_dir,
    tokenizer=make_tokenizer(load_bpe_model(args.tokenizer_model_path))
)

