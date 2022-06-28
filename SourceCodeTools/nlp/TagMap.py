class TagMap:
    def __init__(self, labels):
        self.tag_to_code = dict(zip(labels, range(len(labels))))
        self.labels = labels

    def __getitem__(self, item):
        return self.tag_to_code[item]

    def inverse(self, item):
        return self.labels[item]

    def get(self, key, default=0):
        if key in self.tag_to_code:
            return self.tag_to_code[key]
        else:
            return default

    def __len__(self):
        return len(self.labels)


def tag_map_from_sentences(sentences):
    """
    Map tags to an integer values
    :param sentences: list of tags for sentences
    :return: mapping from tags to integers and mappting from integers to tags
    """
    tags = set()

    # find unique tags
    for s in sentences:
        if s is None:
            continue
        tags.update(set(s))

    return TagMap(list(tags))
    # #
    # # map tags to a contiguous index
    # tagmap = dict(zip(tags, range(len(tags))))
    #
    # aid, iid = zip(*tagmap.items())
    # inv_tagmap = dict(zip(iid, aid))
    #
    # return tagmap, inv_tagmap