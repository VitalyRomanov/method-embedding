import hashlib
import json
import random
import tempfile
from collections import defaultdict
from copy import copy
from pathlib import Path
from math import ceil
from typing import Dict, Optional, Union, Callable

from SourceCodeTools.code.ast.ast_tools import get_declarations
from SourceCodeTools.code.data.file_utils import write_mapping_to_json, read_mapping_from_json
# from SourceCodeTools.models.ClassWeightNormalizer import ClassWeightNormalizer
from SourceCodeTools.nlp import create_tokenizer, TagMap, ValueEncoder, \
    HashingValueEncoder, ValueEmbedder
from SourceCodeTools.nlp.entity import fix_incorrect_tags
from SourceCodeTools.code.annotator_utils import adjust_offsets, biluo_tags_from_offsets
from SourceCodeTools.nlp.entity.utils import overlap
import numpy as np

from nhkv import KVStore
from tqdm import tqdm


def filter_unlabeled(entities, declarations):
    """
    Get a list of declarations that were not mentioned in `entities`
    :param entities: List of entity offsets
    :param declarations: dict, where keys are declaration offsets
    :return: list of declarations that were not mentioned in `entities`
    """
    for_mask = []
    for decl in declarations:
        for e in entities:
            if overlap(decl, e):
                break
        else:
            for_mask.append(decl)
    return for_mask


def print_token_tag(doc, tags):
    for t, tag in zip(doc, tags):
        print(t, "\t", tag)


class SampleEntry(object):
    def __init__(self, id, text, labels=None, category=None, **kwargs):
        self._storage = dict()
        self._storage["id"] = id
        self._storage["text"] = text
        self._storage["labels"] = labels
        self._storage["category"] = category
        self._storage.update(kwargs)

    def __getattr__(self, item):
        storage = object.__getattribute__(self, "_storage")
        if item in storage:
            return storage[item]
        return super().__getattribute__(item)

    def __repr__(self):
        return repr(self._storage)

    def __getitem__(self, item):
        return self._storage[item]

    def __contains__(self, item):
        return item in self._storage

    def keys(self):
        return list(self._storage.keys())


class MapperSpec:
    def __init__(self, field, target_field, encoder, dtype=np.int32, preproc_fn=None, encoder_fn: Union[str, Callable]="seq"):
        self.field = field
        self.target_field = target_field
        self.encoder = encoder
        self.preproc_fn = preproc_fn
        self.dtype = dtype
        self.encoder_fn = encoder_fn


class Batcher:
    def __init__(
            self, data, batch_size: int, seq_len: int,
            wordmap: Dict[str, int], *, tagmap: Optional[TagMap] = None,
            class_weights=False, sort_by_length=True, tokenizer="spacy", no_localization=False,
            cache_dir: Optional[Union[str, Path]] = None, **kwargs
    ):
        self._data = data
        self._batch_size = batch_size
        self._max_seq_len = seq_len
        self._tokenizer = tokenizer
        self._class_weights = None
        self._no_localization = no_localization
        self._nlp = create_tokenizer(tokenizer)
        self._cache_dir = Path(cache_dir) if cache_dir is not None else cache_dir
        self._valid_sentences = 0
        self._filtered_sentences = 0
        self._wordmap = wordmap
        self.tagmap = tagmap
        self.labelmap = None
        self._sort_by_length = sort_by_length
        self._data_ids = set()
        self._batch_generator = None

        self._create_cache()
        self._prepare_data()
        self._create_mappers(**kwargs)

    @property
    def _data_cache_path(self):
        return self._get_cache_location_name("DataCache")

    # @property
    # def _sent_cache_path(self):
    #     return self._get_cache_location_name("SentCache")

    @property
    def _batch_cache_path(self):
        return self._get_cache_location_name("BatchCache")

    @property
    def _length_cache_path(self):
        return self._get_cache_location_name("LengthCache")

    @property
    def _tagmap_path(self):
        return self._cache_dir.joinpath("tagmap.json")

    @property
    def _labelmap_path(self):
        return self._cache_dir.joinpath("labelmap.json")

    @property
    def _unique_tags_and_labels_path(self):
        return self._cache_dir.joinpath("unique_tags_and_labels.json")

    @property
    def _tag_fields(self):
        return ["tags"]

    @property
    def _category_fields(self):
        return ["category"]

    def num_classes(self, how):
        if how == "tags":
            if len(self.tagmap) > 0:
                return len(self.tagmap)
            else:
                raise Exception("There were no tags in the data")
        elif how == "labels":
            if self.labelmap is not None and len(self.labelmap) > 0:
                return len(self.labelmap)
            else:
                raise Exception("There were no labels in the data")
        else:
            raise ValueError(f"Unrecognized category for classes: {how}")

    def _get_version_code(self):
        signature_dict = {
            "tokenizer": self._tokenizer, "max_seq_len": self._max_seq_len, "class_weights": self._class_weights,
            "_no_localization": self._no_localization, "wordmap": self._wordmap, "class": self.__class__.__name__
        }
        if hasattr(self, "_extra_signature_parameters"):
            if hasattr(self, "_extra_signature_parameters_ignore_list"):
                signature_dict.update(
                    {
                        key: val for key, val in self._extra_signature_parameters.items()
                        if key not in self._extra_signature_parameters_ignore_list
                    }
                )
            else:
                signature_dict.update(self._extra_signature_parameters)
        defining_parameters = json.dumps(signature_dict)
        return self._compute_text_id(defining_parameters)

    def _get_cache_location_name(self, cache_name):
        self._check_cache_dir()
        return str(self._cache_dir.joinpath(cache_name))

    def _check_cache_dir(self):
        if not hasattr(self, "_cache_dir") or self._cache_dir is None:
            raise Exception("Cache directory location has not been specified yet")

    def _create_cache(self):
        if self._cache_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory()
            self._cache_dir = Path(self._tmp_dir.name)

        self._cache_dir = self._cache_dir.joinpath(f"{self.__class__.__name__}{self._get_version_code()}")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._data_cache = KVStore(self._data_cache_path)  # dc.Cache(self._data_cache_path)
        self._length_cache = KVStore(self._length_cache_path)
        self._batch_cache = KVStore(self._batch_cache_path)

    @staticmethod
    def _compute_text_id(text):
        return abs(int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)) % 1152921504606846976

    def _prepare_record(self, id_, text, annotations):
        extra = copy(annotations)
        labels = extra.pop("entities")
        extra.update(self._prepare_tokenized_sent((text, annotations)))
        entry = SampleEntry(id=id_, text=text, labels=labels, **extra)
        return entry

    def _update_unique_tags_and_labels(self, unique_tags_and_labels):
        if self._unique_tags_and_labels_path.is_file():
            existing = read_mapping_from_json(self._unique_tags_and_labels_path)
            for field in unique_tags_and_labels:
                if field in existing:
                    existing[field] = set(existing[field])
                    existing[field].update(unique_tags_and_labels[field])
                else:
                    existing[field] = unique_tags_and_labels[field]

            unique_tags_and_labels = existing

        for field in unique_tags_and_labels:
            unique_tags_and_labels[field] = list(unique_tags_and_labels[field])

        write_mapping_to_json(unique_tags_and_labels, self._unique_tags_and_labels_path)

    def _prepare_data(self):
        data_edited = False
        length_edited = True

        unique_tags_and_labels = defaultdict(set)

        def iterate_tags(record, field):
            for label in record[field]:
                yield label

        for text, annotations in tqdm(self._data, desc="Scanning data"):
            id_ = self._compute_text_id(text)
            self._data_ids.add(id_)
            if id_ not in self._length_cache:
                length_edited = True
                if id_ not in self._data_cache:
                    data_edited = True
                    self._data_cache[id_] = (text, annotations)
                entry = self._prepare_record(id_, text, annotations)

                for tag_field in self._tag_fields:
                    unique_tags_and_labels[tag_field].update(set(iterate_tags(entry, tag_field)))

                for cat_field in self._category_fields:
                    cat = entry[cat_field]
                    if cat is not None:
                        unique_tags_and_labels[cat_field].add(entry.category)

                self._length_cache[id_] = len(entry.tokens)

        self._update_unique_tags_and_labels(unique_tags_and_labels)

        if data_edited:
            self._data_cache.save()
        if length_edited:
            self._length_cache.save()

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

    def _get_adjustment(self, doc, offsets):
        if hasattr(doc, "requires_offset_adjustment") and doc.requires_offset_adjustment:
            adjusted_offsets = doc.adjust_offsets(offsets)
            tokens_for_biluo_alignment = doc.get_tokens_for_alignment()
        else:
            adjusted_offsets = offsets
            tokens_for_biluo_alignment = doc
        return tokens_for_biluo_alignment, adjusted_offsets
        # if hasattr(doc, "tokens_for_biluo_alignment"):
        #     entity_adjustment_amount = doc.adjustment_amount
        #     tokens_for_biluo_alignment = doc.tokens_for_biluo_alignment
        # else:
        #     entity_adjustment_amount = 0
        #     tokens_for_biluo_alignment = doc
        # return entity_adjustment_amount, tokens_for_biluo_alignment

    def _biluo_tags_from_offsets(self, doc, tags, check_localization_parameter=False):
        if check_localization_parameter is True:
            no_localization = self._no_localization
        else:
            no_localization = False
        tokens_for_biluo_alignment, adjusted_offsets = self._get_adjustment(doc, tags)
        ents_tags = biluo_tags_from_offsets(
            tokens_for_biluo_alignment, adjusted_offsets,
            no_localization
        )
        fix_incorrect_tags(ents_tags)
        return ents_tags

    def _parse_additional_tags(self, text, annotations, doc, parsed):
        return {}

    def get_record_with_id(self, id_):
        if id_ not in self._data_cache:
            raise KeyError("Record with such id is not found")
        text, annotations = self._data_cache[id_]
        return self._prepare_record(id_, text, annotations)
        # return self._data_cache[id]

    def _iterate_record_ids(self):
        return list(self._data_ids)

    def _iterate_sorted_by_length(self, limit_max_length=False):
        ids = list(self._iterate_record_ids())
        ids_length = list(zip(ids, list(map(lambda x: self._length_cache[x], ids))))
        for id_, length in sorted(ids_length, key=lambda x: x[1]):
            if id_ not in self._data_ids or limit_max_length and length >= self._max_seq_len:
                continue
            yield id_
            # text, annotations = self._get_record_with_id(id_)
            # yield self._prepare_record(id_, text, annotations)

    def _iterate_records(self, limit_max_length=False, shuffle=False):
        ids = self._iterate_record_ids()
        if shuffle:
            random.shuffle(ids)
        for id_ in ids:
            if limit_max_length and self._length_cache[id_] >= self._max_seq_len:
                continue
            yield id_
            # text, annotations = self._get_record_with_id(id_)
            # yield self._prepare_record(id_, text, annotations)

    def _create_mappers(self, **kwargs):
        self._mappers = []
        self._create_wordmap_encoder()
        self._create_tagmap_encoder()
        self._create_category_encoder()
        self._create_additional_encoders()

    def _create_additional_encoders(self):
        pass

    def _create_category_encoder(self, **kwargs):
        if self.labelmap is None:
            if self._labelmap_path.is_file():
                labelmap = TagMap.load(self._labelmap_path)
            else:
                unique_tags_and_labels = read_mapping_from_json(self._unique_tags_and_labels_path)
                if "category" not in unique_tags_and_labels:
                    return
                labelmap = TagMap(
                    unique_tags_and_labels["category"]
                )
                labelmap.save(self._labelmap_path)

            self.labelmap = labelmap

        self._mappers.append(
            MapperSpec(field="category", target_field="label", encoder=self.labelmap, encoder_fn="item")
        )

    def _create_tagmap_encoder(self):
        if self.tagmap is None:
            if self._tagmap_path.is_file():
                tagmap = TagMap.load(self._tagmap_path)
            else:
                unique_tags_and_labels = read_mapping_from_json(self._unique_tags_and_labels_path)
                if "tags" not in unique_tags_and_labels:
                    return
                tagmap = TagMap(
                    unique_tags_and_labels["tags"]
                )
                tagmap.set_default(tagmap._value_to_code["O"])
                tagmap.save(self._tagmap_path)

            self.tagmap = tagmap

        self._mappers.append(
            MapperSpec(field="tags", target_field="tags", encoder=self.tagmap)
        )
        # self.tagmap = tagmap
        # self.tagpad = self.tagmap["O"]

    def _create_wordmap_encoder(self):
        wordmap_enc = ValueEncoder(value_to_code=self._wordmap)
        wordmap_enc.set_default(len(self._wordmap))
        self._mappers.append(
            MapperSpec(field="tokens", target_field="tok_ids", encoder=wordmap_enc)
        )

    # @lru_cache(maxsize=200000)
    def _encode_for_batch(self, record):

        if record.id in self._batch_cache:
            return self._batch_cache[record.id]

        def encode_seq(seq, encoder, pad, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            blank = np.ones((self._max_seq_len,), dtype=np.int32) * pad
            encoded = np.array([encoder[preproc_fn(w)] for w in seq], dtype=np.int32)
            blank[0:min(encoded.size, self._max_seq_len)] = encoded[0:min(encoded.size, self._max_seq_len)]
            return blank

        def encode_item(item, encoder, pad=None, preproc_fn=None):
            if preproc_fn is None:
                def preproc_fn(x):
                    return x
            encoded = np.array(encoder[preproc_fn(item)], dtype=np.int32)
            return encoded

        output = {}

        for mapper in self._mappers:
            if mapper.field in record:
                if mapper.encoder_fn == "item":
                    enc_fn = encode_item
                elif mapper.encoder_fn == "seq":
                    enc_fn = encode_seq
                else:
                    enc_fn = mapper.encoder_fn
                assert isinstance(enc_fn, Callable), "encoder_fn should be either `item`/`seq` or a callable"

                encoded = enc_fn(
                    record[mapper.field], encoder=mapper.encoder, pad=mapper.encoder.default,
                    preproc_fn=mapper.preproc_fn
                ).astype(mapper.dtype)

                if mapper.dtype is not None:
                    output[mapper.target_field] = encoded.astype(mapper.dtype)
                else:
                    output[mapper.target_field] = encoded

        num_tokens = len(record.tokens)

        output["lens"] = np.array(num_tokens if num_tokens < self._max_seq_len else self._max_seq_len, dtype=np.int32)
        output["id"] = record.id

        self._batch_cache[record.id] = output
        self._batch_cache.save()

        return output

    def format_batch(self, batch):
        fbatch = defaultdict(list)

        for sent in batch:
            for key, val in sent.items():
                fbatch[key].append(val)

        max_len = max(fbatch["lens"])

        batch_o = {}

        for field, items in fbatch.items():
            if field == "lens" or field == "label":
                batch_o[field] = np.array(items, dtype=np.int32)
            elif field == "id":
                batch_o[field] = np.array(items, dtype=np.int64)
            elif field == "tokens" or field == "replacements":
                batch_o[field] = items
            else:
                batch_o[field] = np.stack(items)[:, :max_len]

        return batch_o

        # return {
        #     key: np.stack(val)[:,:max_len] if key != "lens" and key != "replacements" and key != "tokens"
        #     else (np.array(val, dtype=np.int32) if key == "lens" else np.array(val)) for key, val in fbatch.items()}

    def generate_batches(self):
        batch = []
        if self._sort_by_length:
            records = self._iterate_sorted_by_length(limit_max_length=True)
        else:
            records = self._iterate_records(limit_max_length=True, shuffle=False)

        for id_ in records:
            if id_ in self._batch_cache:
                batch.append(self._batch_cache[id_])
            else:
                batch.append(self._encode_for_batch(self.get_record_with_id(id_)))
            if len(batch) >= self._batch_size:
                yield self.format_batch(batch)
                batch.clear()

        # for sent in records:
        #     batch.append(self._encode_for_batch(sent))
        #     if len(batch) >= self._batch_size:
        #         yield self.format_batch(batch)
        #         batch = []
        if len(batch) > 0:
            yield self.format_batch(batch)
        # yield self.format_batch(batch)

    def __iter__(self):
        self._batch_generator = self.generate_batches()
        return self

    def __next__(self):
        if self._batch_generator is None:
            raise StopIteration()
        return next(self._batch_generator)

    def __len__(self):
        total_valid = 0
        for id_ in self._iterate_record_ids():
            length = self._length_cache[id_]
            if length < self._max_seq_len:
                total_valid += 1
        return int(ceil(total_valid / self._batch_size))


class PythonBatcher(Batcher):
    def __init__(
            self, *args, graphmap, element_hash_size, mask_unlabeled_declarations=False, **kwargs
    ):
        self._element_hash_size = element_hash_size
        self._graphmap = graphmap
        self._mask_unlabeled_declarations = mask_unlabeled_declarations

        self._extra_signature_parameters = {
            "element_hash_size": element_hash_size,
            "mask_unlabeled_declarations": mask_unlabeled_declarations,
            "graphmap": self._graphmap if not hasattr(self._graphmap, "signature") else self._graphmap.signature()
        }

        super(PythonBatcher, self).__init__(*args, **kwargs)

    def _parse_additional_tags(self, text, annotations, doc, parsed):

        repl_tags = self._biluo_tags_from_offsets(doc, annotations["replacements"])
        assert len(parsed["tokens"]) == len(repl_tags)

        if self._mask_unlabeled_declarations:
            unlabeled_dec = filter_unlabeled(annotations["entities"], get_declarations(text))
            unlabeled_dec_tags = self._biluo_tags_from_offsets(doc, unlabeled_dec)
            assert len(parsed["tokens"]) == len(unlabeled_dec_tags)
        else:
            unlabeled_dec_tags = ["O"] * len(parsed["tokens"])

        return {
            "replacements": repl_tags,
            "unlabeled_dec": unlabeled_dec_tags
        }

    def _strip_biluo_and_try_cast_to_int(self, graph_tag):
        if "-" in graph_tag:
            graph_tag = graph_tag.split("-")[-1]
        try:
            graph_tag = int(graph_tag)
        except:
            pass
        return graph_tag

    def _create_graph_encoder(self):
        if self._graphmap is not None:
            graphmap_enc = ValueEncoder(value_to_code=self._graphmap)
            graphmap_enc.set_default(len(self._graphmap))
            self._mappers.append(
                MapperSpec(
                    field="replacements", target_field="graph_ids", encoder=graphmap_enc,
                    preproc_fn=self._strip_biluo_and_try_cast_to_int, dtype=np.int32
                )
            )
            self.graphpad = graphmap_enc.default
        else:
            self.graphpad = 0

    def _create_additional_encoders(self):

        def suffix(word):
            return word[-3:]

        def prefix(word):
            return word[:3]

        self._mappers.append(
            MapperSpec(
                field="tokens", target_field="prefix", encoder=HashingValueEncoder(self._element_hash_size),
                preproc_fn=prefix, dtype=np.int32
            )
        )

        self._mappers.append(
            MapperSpec(
                field="tokens", target_field="suffix", encoder=HashingValueEncoder(self._element_hash_size),
                preproc_fn=suffix, dtype=np.int32
            )
        )

        self._create_graph_encoder()

        class NoMaskEncoder(ValueEncoder):
            def __init__(self, default=False, *args, **kwargs):
                super(NoMaskEncoder, self).__init__(default=default)

            def _initialize(self, values, value_to_code):
                pass

            def __getitem__(self, item):
                if item == "O":
                    return False
                else:
                    return True

            def get(self, key, default=0):
                return self.__getitem__(key)

        self._mappers.append(
            MapperSpec(
                field="tags", target_field="no_loc_mask", encoder=NoMaskEncoder(),
                preproc_fn=self._strip_biluo_and_try_cast_to_int, dtype=np.bool
            )
        )

        class OMaskEncoder(NoMaskEncoder):
            def __init__(self, default=True, *args, **kwargs):
                super(OMaskEncoder, self).__init__(default=default)

            def __getitem__(self, item):
                if item == "O":
                    return True
                else:
                    return False

        self._mappers.append(
            MapperSpec(
                field="unlabeled_dec", target_field="hide_mask", encoder=OMaskEncoder(),
                preproc_fn=self._strip_biluo_and_try_cast_to_int, dtype=np.bool
            )
        )


class PythonBatcherWithGraphEmbeddings(PythonBatcher):
    def __init__(self, *args, **kwargs):
        # self._extra_signature_parameters_ignore_list = ["graphmap"]
        super(PythonBatcherWithGraphEmbeddings, self).__init__(*args, **kwargs)

    def _create_graph_encoder(self):
        if self._graphmap is not None:

            def encode_feat_vec_seq(seq, encoder, pad, preproc_fn=None):
                if preproc_fn is None:
                    def preproc_fn(x):
                        return x
                encoded = np.array([encoder[preproc_fn(w)] for w in seq], dtype=np.float32)
                blank = np.ones((self._max_seq_len, encoded.shape[1]), dtype=np.float32) * pad
                blank[0:min(encoded.shape[0], self._max_seq_len), :] = encoded[0:min(encoded.size, self._max_seq_len), :]
                return blank

            graphmap_enc = ValueEmbedder(
                value_to_code=self._graphmap,
                default=np.zeros(shape=(self._graphmap.n_dims,))
            )

            self._mappers.append(
                MapperSpec(
                    field="replacements", target_field="graph_embs", encoder=graphmap_enc,
                    preproc_fn=self._strip_biluo_and_try_cast_to_int, dtype=np.float32,
                    encoder_fn=encode_feat_vec_seq
                )
            )
            self.graphpad = graphmap_enc.default
        else:
            self.graphpad = 0
