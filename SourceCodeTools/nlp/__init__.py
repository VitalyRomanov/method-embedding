import hashlib


def token_hasher(token: str, buckets: int):
    return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16) % buckets
