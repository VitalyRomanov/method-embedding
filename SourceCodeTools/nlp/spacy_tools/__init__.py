import logging


from spacy.gold import biluo_tags_from_offsets, offsets_from_biluo_tags
from spacy.cli.init_model import add_vectors as spacy_add_vectors



def isvalid(nlp, text, ents):
    """
    Verify that tokens and entity spans are aligned
    :param nlp: spacy tokenizer
    :param text: text to tokenize
    :param ents: list of entities in format (start, end, entity)
    :return: true if entities and tokens are aligned
    """
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, ents)
    if "-" in tags:
        return False
    else:
        return True


def deal_with_incorrect_offsets(sents, nlp):

    for ind in range(len(sents)):
        doc = nlp(sents[ind][0])

        tags = biluo_tags_from_offsets(doc, sents[ind][1]['entities'])

        sents[ind][1]['entities'] = offsets_from_biluo_tags(doc, tags)

        if "replacements" in sents[ind][1]:
            tags = biluo_tags_from_offsets(doc, sents[ind][1]['replacements'])
            while "-" in tags:
                tags[tags.index("-")] = "O"
            sents[ind][1]['replacements'] = offsets_from_biluo_tags(doc, tags)

    return sents


def add_vectors(nlp, vectors_loc, truncate_vectors=0, prune_vectors=-1, vectors_name=None):
    spacy_add_vectors(nlp, vectors_loc, truncate_vectors, prune_vectors, vectors_name)
    vec_added = len(nlp.vocab.vectors)
    lex_added = len(nlp.vocab)
    logging.info(
        "Successfully compiled vocab",
        "{} entries, {} vectors".format(lex_added, vec_added),
    )
