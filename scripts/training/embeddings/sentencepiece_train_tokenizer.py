import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser(description='Train sentencepiece tokenizer')
parser.add_argument('--code_path', dest='code_path', default=None,
                    help='')
parser.add_argument('--vocab_size', "-v", dest='vocab_size', default=100000,
                    help='')
parser.add_argument('--model_type', dest='model_type', default='bpe',
                    help='')
parser.add_argument('--input_format', dest='input_format', default='text',
                    help='')
parser.add_argument('--apicall_path', dest='apicall_path', default=None,
                    help='')
parser.add_argument('--out_path', dest='out_path', default=None,
                    help='')


args = parser.parse_args()


spm.SentencePieceTrainer.train(input=args.code_path,
                               input_format=args.input_format,
                               model_prefix='sentencepiece' + "_" + args.model_type,
                               vocab_size=args.vocab_size,
                               model_type=args.model_type,
                               character_coverage=1.0,
                               input_sentence_size=1000000,
                               shuffle_input_sentence=True)
