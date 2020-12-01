
class ClassWeightNormalizer:
    def __init__(self):
        self.class_counter = None

    def init(self, classes):
        import itertools
        from collections import Counter
        self.class_counter = Counter(c for c in itertools.chain.from_iterable(classes))
        total_count = sum(self.class_counter.values())
        self.class_weights = {key: total_count / val for key, val in self.class_counter.items()}

    def __getitem__(self, item):
        return self.class_weights[item]

    def get(self, item, default):
        return self.class_weights.get(item, default)