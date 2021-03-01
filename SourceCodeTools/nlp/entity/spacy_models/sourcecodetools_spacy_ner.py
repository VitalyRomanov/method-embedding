from __future__ import unicode_literals, print_function
import spacy
import sys, json, os
import pickle

from SourceCodeTools.nlp import create_tokenizer
from SourceCodeTools.nlp.entity.utils.data import read_data
from spacy.gold import biluo_tags_from_offsets

from spacy.gold import GoldParse
from spacy.scorer import Scorer
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

from SourceCodeTools.nlp.entity.utils.spacy_tools import isvalid, add_vectors


class SpacyNERTrainer:
    def __init__(self):
        pass


def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        doc_gold_text = ner_model.make_doc(input_)
        gold = GoldParse(doc_gold_text, entities=annot['entities'])
        pred_value = ner_model(input_)
        scorer.score(pred_value, gold)
    return {key: scorer.scores[key] for key in ['ents_p', 'ents_r', 'ents_f', 'ents_per_type']}
    # return scorer.scores['ents_per_type']


def train_spacy_model(train_data, test_data, model, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    nlp = model

    ent_types = []
    for _, e in train_data:
        ee = [ent[2] for ent in e['entities']]
        ent_types += ee

    for text, ent in train_data:
        doc = nlp(text)
        entities = ent['entities']
        tags = biluo_tags_from_offsets(doc, entities)

        # # if "-" in tags:
        # print(text)
        # print(entities, tags)
        # for t in doc:
        #     print(t, tags[t.i])
        # print("\n\n\n")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        # if model is None:
        nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_data)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print(f"{itn}:")
            print("\tLosses", losses)
            score = evaluate(nlp, test_data)
            if not os.path.isdir("models"):
                os.mkdir("models")
            nlp.to_disk(os.path.join("models", f"model_{itn}"))
            print("\t", score)

    # test the trained model

    # print(score)
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # # save model to output directory
    # if output_dir is not None:
    #     output_dir = Path(output_dir)
    #     if not output_dir.exists():
    #         output_dir.mkdir()
    #     nlp.to_disk(output_dir)
    #     print("Saved model to", output_dir)
    #
    #     # test the saved model
    #     print("Loading from", output_dir)
    #     nlp2 = spacy.load(output_dir)
    #
    #     for text, _ in test_data:
    #         doc = nlp2(text)
    #         print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #         print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])



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


    train_data, test_data = read_data(open(args.data_path, "r").readlines(), random_seed=args.seed)

    if args.model_path is not None:
        model = spacy.load(args.model_path)
    else:
        if args.vectors is not None:
            model = create_tokenizer("spacy_bpe", bpe_path=args.bpe)
            add_vectors(model, args.vectors)
        else:
            raise Exception("You should provide either an initialized spacy model or pretrained vectors")


    train_spacy_model(train_data, test_data, model=model, output_dir=args.output_model, n_iter=args.epochs)


if __name__ == "__main__":
    main()