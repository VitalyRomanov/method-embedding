def create_subword_tokenizer(lang, vs):
    from pathlib import Path
    from bpemb.util import sentencepiece_load, http_get
    import re

    def _load_file(file, archive=False):
        cache_dir = Path.home() / Path(".cache/bpemb")
        archive_suffix = ".tar.gz"
        base_url = "https://nlp.h-its.org/bpemb/"
        cached_file = Path(cache_dir) / file
        if cached_file.exists():
            return cached_file
        suffix = archive_suffix if archive else ""
        file_url = base_url + file + suffix
        print("downloading", file_url)
        return http_get(file_url, cached_file, ignore_tardir=True)
    model_file = "{lang}/{lang}.wiki.bpe.vs{vs}.model".format(lang=lang, vs=vs)
    model_file = _load_file(model_file)
    spm = sentencepiece_load(model_file)
    return lambda text: spm.EncodeAsPieces(re.sub(r"\d", "0", text.lower()))

def load_bpe_model(path):
    from sentencepiece import SentencePieceProcessor
    spm = SentencePieceProcessor()
    spm.Load(path)
    if spm.Load(path):
        return spm
    else:
        raise Exception("Error loading model")



def make_tokenizer(bpe):
    return lambda text: bpe.EncodeAsPieces(text)