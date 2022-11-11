import numpy as np

from SourceCodeTools.nlp.batchers import PythonBatcher


class HybridBatcher(PythonBatcher):
    def __init__(self, *args, **kwargs):
        super(HybridBatcher, self).__init__(*args, **kwargs)

    def _prepare_tokenized_sent(self, sent):
        text, annotations = sent

        doc = self._nlp(text)
        ents = annotations['entities']

        tokens = doc
        try:
            tokens = [t.text for t in tokens]
        except:
            pass

        ents_tags = self._biluo_tags_from_offsets(doc, ents, check_localization_parameter=True)
        assert len(tokens) == len(ents_tags)

        output = {
            "tokens": tokens,
            "tags": ents_tags
        }

        output.update(self._parse_additional_tags(text, annotations, doc, output))

        return output

    def _encode_for_batch(self, record):

        if record.id in self._batch_cache:
            return self._batch_cache[record.id]

        def encode(seq, encoder, pad, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            blank = np.ones((self._max_seq_len,), dtype=np.int32) * pad
            encoded = np.array([encoder[preproc_fn(w)] for w in seq], dtype=np.int32)
            blank[0:min(encoded.size, self._max_seq_len)] = encoded[0:min(encoded.size, self._max_seq_len)]
            return blank

        def encode_label(item, encoder, pad=None, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            encoded = np.array(encoder[preproc_fn(item)], dtype=np.int32)
            return encoded

        output = {}

        for mapper in self._mappers:
            if mapper.field in record:
                if mapper.target_field == "label":
                    enc_fn = encode_label
                else:
                    enc_fn = encode

                output[mapper.target_field] = enc_fn(
                    record[mapper.field], encoder=mapper.encoder, pad=mapper.encoder.default,
                    preproc_fn=mapper.preproc_fn
                ).astype(mapper.dtype)

        num_tokens = len(record.tokens)

        output["lens"] = np.array(num_tokens if num_tokens < self._max_seq_len else self._max_seq_len, dtype=np.int32)
        output["id"] = record.id

        self._batch_cache[record.id] = output
        self._batch_cache.save()

        return output