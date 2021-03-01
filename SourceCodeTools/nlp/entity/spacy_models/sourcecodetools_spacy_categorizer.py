#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import thinc.extra.datasets

from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.utils import get_unique_entities
from SourceCodeTools.nlp.entity.utils.data import read_data


import spacy
from spacy.util import minibatch, compounding

from SourceCodeTools.nlp.spacy_tools import add_vectors


def train_spacy_categorizer(train_data, test_data, model=None, output_dir=None, n_iter=20):
    nlp = model

    cats = get_unique_entities(train_data, "categories")

    def format_cats(data, all_cats):
        # new_data = []
        # for text, annotations in data:
        #     new_ann = {f"{cat}": cat in annotations["categories"] for cat in all_cats}
        #     new_data.append(text, new_ann)
        # return new_data
        return [(text, {f"{cat}": cat in annotations["categories"] for cat in all_cats}) for text, annotations in data]

    train_data = format_cats(train_data, cats)
    test_data = format_cats(test_data, cats)

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"exclusive_classes": True, "architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    for cat in cats:
        textcat.add_label(cat)

    train_texts, train_cats = zip(*train_data)
    test_texts, test_cats = zip(*test_data)
    print(
        "Using {} examples ({} training, {} evaluation)".format(
            len(train_texts) + len(test_texts), len(train_texts), len(test_texts)
        )
    )
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))
    test_data = list(zip(test_texts, [{"cats": cats} for cats in test_cats]))

    # get names of other pipes to disable them during training
    pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            # scores = evaluate(nlp, test_data)
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate(textcat, nlp, test_data)
            print(scores)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


# def load_data(limit=0, split=0.8):
#     """Load data from the IMDB dataset."""
#     # Partition off part of the train data for evaluation
#     train_data, _ = thinc.extra.datasets.imdb()
#     random.shuffle(train_data)
#     train_data = train_data[-limit:]
#     texts, labels = zip(*train_data)
#     cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
#     split = int(len(train_data) * split)
#     return (texts[:split], cats[:split]), (texts[split:], cats[split:])
from spacy.gold import GoldParse
from spacy.scorer import Scorer

def evaluate(textcat, ner_model, examples):
    scorer = Scorer(pipeline=ner_model.pipeline)
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        # textcat_doc = textcat(doc_gold_text)
        cats = {key: 1. if key in annot['cats'] else 0. for key in textcat.labels}
        gold = GoldParse(doc_gold_text, cats=cats)
        pred_value = textcat(ner_model(input_))
        scorer.score(pred_value, gold)
    return {key: scorer.scores[key] for key in ['textcat_score', 'textcats_per_cat']}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-m", dest="model_path", default=None)
    parser.add_argument("--vectors", "-v", dest="vectors", default=None)
    parser.add_argument("data_path")
    parser.add_argument("--output_model", "-o", dest="output_model", default="spacy-typing-ner")
    parser.add_argument("--epochs", "-e", dest="epochs", default=90, type=int)
    parser.add_argument("--seed", "-s", dest="seed", default=42, type=int, help="Seed for random dataset split")
    parser.add_argument("--bpe", dest="bpe", default=None, type=str, help="")
    args = parser.parse_args()


    train_data, test_data = read_data(open(args.data_path, "r").readlines(), include_only="categories", random_seed=args.seed)

    if args.model_path is not None:
        model = spacy.load(args.model_path)
    else:
        if args.vectors is not None:
            model = create_tokenizer("spacy_bpe", bpe_path=args.bpe)
            add_vectors(model, args.vectors)
        else:
            raise Exception("You should provide either an initialized spacy model or pretrained vectors")


    train_spacy_categorizer(train_data, test_data, model=model, output_dir=args.output_model, n_iter=args.epochs)


if __name__ == "__main__":
    main()
