from __future__ import unicode_literals, print_function

import json
import os
import pickle

import tensorflow as tf

from SourceCodeTools.nlp.entity import parse_biluo
from SourceCodeTools.nlp.entity.tf_models.tf_model import TypePredictor
from SourceCodeTools.nlp.entity.type_prediction import PythonBatcher, load_pkl_emb
from SourceCodeTools.nlp.entity.utils.data import read_data


def logits_to_annotations(predicted, lengths, tagmap):
    mask = tf.sequence_mask(lengths, predicted.shape[1])
    argmax = predicted
    nulled = argmax * tf.cast(mask, dtype=tf.int64)

    out_tags = tf.cast(tf.logical_not(mask), dtype=tf.int64) * tagmap['O']
    tag_labels = tf.maximum(nulled, out_tags)  # out_tags will be chosen for masked positions

    ents = []
    for i in range(tag_labels.shape[0]):
        line = tag_labels[i].numpy()
        tag_line = [tagmap.inverse(v) for v in line]
        ent = parse_biluo(tag_line)
        ents.append(ent)

    return ents


def read_config(config_path):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    config = config["DEFAULT"]
    return {key: config[key] for key in config}


def predict_one(model, input, tagmap):
    return logits_to_annotations(
            tf.math.argmax(model(
                token_ids=input['tok_ids'], prefix_ids=input['prefix'], suffix_ids=input['suffix'],
                graph_ids=input['graph_ids'], training=False
            ), axis=-1),
            input['lens'],
            tagmap
        )


def apply_to_dataset(data, Batcher, Model, graph_emb_path=None, word_emb_path=None, checkpoint_path=None, batch_size=1):
    graph_emb = load_pkl_emb(graph_emb_path)
    word_emb = load_pkl_emb(word_emb_path)

    with open(os.path.join(checkpoint_path, "params.json"), "r") as json_params:
        params = json.loads(json_params.read().strip())

    tagmap = pickle.load(open(os.path.join(checkpoint_path, "tag_types.pkl"), "rb"))
    # params = read_config(checkpoint_path)

    seq_len = params.pop("seq_len")
    suffix_prefix_buckets = params.pop("suffix_prefix_buckets")

    data_batcher = Batcher(
        data, batch_size, seq_len=seq_len, wordmap=word_emb.ind, graphmap=graph_emb.ind, tagmap=tagmap,
        mask_unlabeled_declarations=True, class_weights=False, element_hash_size=suffix_prefix_buckets
    )

    params.pop("train_losses")
    params.pop("test_losses")
    params.pop("train_f1")
    params.pop("test_f1")
    params.pop("epochs")
    params.pop("learning_rate")
    params.pop("learning_rate_decay")

    model = Model(
        word_emb, graph_emb, train_embeddings=False, num_classes=len(tagmap), seq_len=seq_len,
        suffix_prefix_buckets=suffix_prefix_buckets, **params
    )

    model.load_weights(os.path.join(checkpoint_path, "checkpoint"))

    all_true = []
    all_estimated = []

    for batch in data_batcher:
        true_annotations = logits_to_annotations(batch['tags'], batch['lens'], tagmap)
        est_annotations = predict_one(model, batch, tagmap)
        all_true.extend(true_annotations)
        all_estimated.extend(est_annotations)

    from SourceCodeTools.nlp.entity.entity_render import render_annotations

    html = render_annotations(zip(data, all_estimated, all_true))
    with open("render.html", "w") as render:
        render.write(html)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', dest='data_path', default=None,
                        help='Path to the file with nodes')
    parser.add_argument('--graph_emb_path', dest='graph_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('--word_emb_path', dest='word_emb_path', default=None,
                        help='Path to the file with edges')
    parser.add_argument('checkpoint_path', default=None, help='')

    args = parser.parse_args()

    train_data, test_data = read_data(
        open(args.data_path, "r").readlines(), normalize=True, allowed=None, include_replacements=True,
        include_only="entities",
        min_entity_count=5
    )

    apply_to_dataset(
        test_data, PythonBatcher, TypePredictor, graph_emb_path=args.graph_emb_path, word_emb_path=args.word_emb_path,
        checkpoint_path=args.checkpoint_path, batch_size=10
    )
