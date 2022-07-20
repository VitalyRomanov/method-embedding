from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.embed.fasttext import train_wor2vec
import argparse

parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('tokenizer_name', type=str, default=None, help='Tokenizer name, see SourceCodeTools.nlp.tokenizers')
parser.add_argument('input_file', type=str, default=None, help='Path to corpus')
parser.add_argument('output_dir', type=str, default=None, help='Output saving directory')
parser.add_argument('--emb_size', type=int, default=100, help='')
args = parser.parse_args()

train_wor2vec(
    corpus_path=args.input_file,
    output_path=args.output_dir,
    tokenizer=create_tokenizer(args.tokenizer_name),
    emb_size=args.emb_size
)

