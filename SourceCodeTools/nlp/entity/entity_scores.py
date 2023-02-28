from SourceCodeTools.nlp.entity import parse_biluo


def compute_precision_recall_f1(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


def localized_f1(pred_spans, true_spans, eps=1e-8):

    tp = 0.
    fp = 0.
    fn = 0.

    for pred, true in zip(pred_spans, true_spans):
        if true != "O":
            if true == pred:
                tp += 1
            else:
                if pred == "O":
                    fn += 1
                else:
                    fp += 1

    return compute_precision_recall_f1(tp, fp, fn)


def span_f1(pred_spans, true_spans, eps=1e-8):
    tp = len(pred_spans.intersection(true_spans))
    fp = len(pred_spans - true_spans)
    fn = len(true_spans - pred_spans)

    return compute_precision_recall_f1(tp, fp, fn)


def token_spans_from_prediction(predictions, tagmap):
    pred_biluo = [tagmap.inverse(p) for p in predictions]
    return parse_biluo(pred_biluo)


def entity_scorer(pred, labels, tagmap, no_localization=False, eps=1e-8):
    """
    Compute f1 score, precision, and recall from BILUO labels
    :param pred: predicted BILUO labels
    :param labels: ground truth BILUO labels
    :param tagmap:
    :param eps:
    :return:
    """
    # TODO
    # the scores can be underestimated because ground truth does not contain all possible labels
    # this results in higher reported false alarm rate
    pred_biluo = [tagmap.inverse(p) for p in pred]
    labels_biluo = [tagmap.inverse(p) for p in labels]

    if not no_localization:
        pred_spans = set(parse_biluo(pred_biluo))
        true_spans = set(parse_biluo(labels_biluo))

        precision, recall, f1 = span_f1(pred_spans, true_spans, eps=eps)
    else:
        precision, recall, f1 = localized_f1(pred_biluo, labels_biluo, eps=eps)

    return precision, recall, f1