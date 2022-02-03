from os import urandom


class IdentifierPool:
    def __init__(self):
        self._used_identifiers = set()

    @staticmethod
    def _get_candidate():
        return "0x" + str(urandom(8).hex())

    def get_new_identifier(self):
        candidate = self._get_candidate()
        while candidate in self._used_identifiers:
            candidate = self._get_candidate()
        self._used_identifiers.add(candidate)
        return candidate


class IntIdentifierPool(IdentifierPool):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_candidate():
        candidate = str(int(urandom(10).hex(), 16))
        while len(candidate) < 19:
            candidate = str(int(urandom(10).hex(), 16))
        return candidate[:19]