from SourceCodeTools.nlp.embed.bpe import load_bpe_model, make_tokenizer
from SourceCodeTools.nlp.embed.fasttext import train_wor2vec
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('tokenizer_model_path', type=str, default=None, help='Path to sentencepiece tokenizer model')
parser.add_argument('input_file', type=str, default=None, help='Path to corpus')
parser.add_argument('output_dir', type=str, default=None, help='Output saving directory')
parser.add_argument('--emb_size', type=int, default=100, help='')
args = parser.parse_args()

train_wor2vec(
    corpus_path=args.input_file,
    output_path=args.output_dir,
    tokenizer=make_tokenizer(load_bpe_model(args.tokenizer_model_path)),
    emb_size=args.emb_size
)

