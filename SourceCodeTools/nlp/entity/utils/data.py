from SourceCodeTools.nlp.entity.utils import *

def read_data(
        data, normalize=False, include_replacements=False, allowed=None, min_entity_count=None,
        include_only="entities",
        random_seed=None, train_frac=0.8
):
    """
    Read dataset and split into train and test
    :param data: NER dataset in jsonl format
    :param normalize: Optional. Whether to apply normalization function. See `normalize_entities`.
    :param include_replacements: Optional. Whether to include replacements into the data
    :param allowed: Optional. A list of allowed entities for filtering. Entities not on the list are replaced
        with "Other".
    :param min_entity_count: Optional. Set minimum frequency for entities. Entities wit hcount less than this will be
        renamed `Other`.
    :param include_only: Optional: entities|categories. Default: entities. Whether to read dataset for NER
        or classification.
    :param random_seed: Optional. Set this to enforce repeatable data splits.
    :param train_frac: Optional. Default: 0.8. Proportion of train set.
    :return: train and test sets
    """
    import random

    train_data = []
    entities_in_dataset = []

    assert include_only in {"entities", "categories"}

    for line in data:
        entry = json.loads(line)
        # format for entry is (text, annotation_dict)
        train_data.append(format_record(entry, include_only, allowed))

        if len(train_data[-1][1][include_only]) == 0:
            train_data.pop(-1)
            continue

        if include_replacements:
            if "replacements" in entry:
                # [1] is for annotation_dict
                train_data[-1][1]['replacements'] = resolve_repeats(entry['replacements'])

        if normalize:
            train_data[-1][1][include_only] = normalize_entities(train_data[-1][1][include_only])

        # ensure spans are given as integers
        if include_only == "entities":
            train_data[-1][1]['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in train_data[-1][1]['entities']]

        for entity in train_data[-1][1][include_only]:
            entities_in_dataset.append(get_entity_name(entity))

    if random_seed is None:
        logging.info("Splitting dataset randomly")
    else:
        random.seed(random_seed)
        logging.warning(f"Using random seed {random_seed} for dataset split")

    filter_infrequent(
        train_data, entities_in_dataset=Counter(entities_in_dataset),
        field=include_only, min_entity_count=min_entity_count, behaviour="rename"
    )

    random.shuffle(train_data)
    train_size = int(len(train_data) * train_frac)
    train, test = train_data[:train_size], train_data[train_size:]

    entities_in_train = get_unique_entities(train, field=include_only)

    filter_entities(test, field=include_only, allowed=entities_in_train, behaviour="drop")

    assert len(test) > 0, """
    It appears test set is too small. Reasons are:
        - dataset is too small
        - label distribution is such that no labels in test set are present in train set
        - parameter `train_frac` has inadequate value for the given dataset size
    """

    return train, test


def read_json_data(
        data_path, normalize=False, include_replacements=False, allowed=None, min_entity_count=None,
        include_only="entities",
        random_seed=None, train_frac=0.8
):
    """
    Read dataset and split into train and test
    :param data: NER dataset in jsonl format
    :param normalize: Optional. Whether to apply normalization function. See `normalize_entities`.
    :param include_replacements: Optional. Whether to include replacements into the data
    :param allowed: Optional. A list of allowed entities for filtering. Entities not on the list are replaced
        with "Other".
    :param min_entity_count: Optional. Set minimum frequency for entities. Entities wit hcount less than this will be
        renamed `Other`.
    :param include_only: Optional: entities|categories. Default: entities. Whether to read dataset for NER
        or classification.
    :param random_seed: Optional. Set this to enforce repeatable data splits.
    :param train_frac: Optional. Default: 0.8. Proportion of train set.
    :return: train and test sets
    """
    import random



    assert include_only in {"entities", "categories"}

    train_file = data_path.joinpath("type_prediction_dataset_no_default_args_mapped_train.json")
    test_file = data_path.joinpath("type_prediction_dataset_no_default_args_mapped_test.json")

    def read_data_file(path):
        data = []
        entities_in_dataset = []
        for line in open(path, "r"):
            entry = json.loads(line)
            # format for entry is (text, annotation_dict)
            data.append(format_json_record(entry, include_only, allowed))

            if len(data[-1][1][include_only]) == 0:
                data.pop(-1)
                continue

            if include_replacements:
                if "replacements" in entry[1]:
                    # [1] is for annotation_dict
                    data[-1][1]['replacements'] = resolve_repeats(entry[1]['replacements'])

            if normalize:
                data[-1][1][include_only] = normalize_entities(data[-1][1][include_only])

            # ensure spans are given as integers
            if include_only == "entities":
                data[-1][1]['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in data[-1][1]['entities']]

            for entity in data[-1][1][include_only]:
                entities_in_dataset.append(get_entity_name(entity))

        return data, entities_in_dataset

    train_data, entities_in_dataset = read_data_file(train_file)
    test_data, _ = read_data_file(test_file)

    filter_infrequent(
        train_data, entities_in_dataset=Counter(entities_in_dataset),
        field=include_only, min_entity_count=min_entity_count, behaviour="rename"
    )

    entities_in_train = get_unique_entities(train_data, field=include_only)

    filter_entities(test_data, field=include_only, allowed=entities_in_train, behaviour="drop")

    assert len(test_data) > 0, """
    It appears test set is too small. Reasons are:
        - dataset is too small
        - label distribution is such that no labels in test set are present in train set
        - parameter `train_frac` has inadequate value for the given dataset size
    """

    return train_data, test_data


def test_read_data():
    test_data = [
        '{"ents": [[23, 31, "int"]], "cats": [], "replacements": [[43, 47, "16"], [48, 53, "5"], [56, 64, "19"], [17, 21, "17"], [8, 16, "2"]], "text": "    def __init__(self, argument):\\n\\n        self.field = argument\\n", "docstrings": ["        \\"\\"\\"\\n        Initialize. \\u0418\\u043d\\u0438\\u0446\\u0438\\u0430\\u043b\\u0438\\u0437\\u0430\\u0446\\u0438\\u044f\\n        :param argument:\\n        \\"\\"\\""]}\n',
        '{"ents": [], "cats": ["str"], "replacements": [[45, 52, "8"], [8, 15, "6"], [16, 20, "34"], [40, 44, "33"]], "text": "    def method1(self) :\\n\\n        return self.method2()\\n", "docstrings": ["        \\"\\"\\"\\n        Call another method. \\u0412\\u044b\\u0437\\u043e\\u0432 \\u0434\\u0440\\u0443\\u0433\\u043e\\u0433\\u043e \\u043c\\u0435\\u0442\\u043e\\u0434\\u0430.\\n        :return:\\n        \\"\\"\\""]}\n',
        '{"ents": [[33, 42, "int"], [64, 73, "str"]], "cats": ["str"], "replacements": [[106, 115, "52"], [64, 73, "52"], [76, 79, "7"], [50, 55, "5"], [33, 42, "50"], [8, 15, "8"], [16, 20, "47"]], "text": "    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2", "docstrings": ["        \\"\\"\\"\\n        Simple operations.\\n        \\u041f\\u0440\\u043e\\u0441\\u0442\\u044b\\u0435 \\u043e\\u043f\\u0435\\u0440\\u0430\\u0446\\u0438\\u0438.\\n        :return:\\n        \\"\\"\\""]}\n',
        '{"ents": [], "cats": ["None"], "replacements": [[17, 22, "13"], [23, 31, "77"], [4, 8, "12"], [32, 39, "6"]], "text": "def main() :\\n    print(instance.method1())\\n", "docstrings": []}\n',
        '{"ents": [[23, 28, "int"]], "cats": [], "replacements": [[17, 21, "111"], [51, 56, "113"], [45, 48, "104"], [40, 44, "110"], [8, 16, "103"]], "text": "    def __init__(self, value):\\n\\n        self.val = value\\n", "docstrings": ["        \\"\\"\\"\\n        Initialize. \\u0418\\u043d\\u0438\\u0446\\u0438\\u0430\\u043b\\u0438\\u0437\\u0430\\u0446\\u0438\\u044f\\n        :param argument:\\n        \\"\\"\\""]}\n',
        '{"ents": [], "cats": ["str"], "replacements": [[17, 21, "137"], [8, 16, "106"], [56, 59, "104"]], "text": "    def __repr__(self) :\\n\\n        return f\\"Number({self.val})\\"", "docstrings": ["        \\"\\"\\"\\n        Return representation\\n        :return: \\u041f\\u043e\\u043b\\u0443\\u0447\\u0438\\u0442\\u044c \\u043f\\u0440\\u0435\\u0434\\u0441\\u0442\\u0430\\u0432\\u043b\\u0435\\u043d\\u0438\\u0435\\n        \\"\\"\\""]}\n',
        '{"ents": [[14, 18, "list"]], "cats": [], "replacements": [[4, 13, "477"]], "text": "def annotated(arg1):\\n    pass\\n", "docstrings": []}\n'
    ]

    train, test = read_data(test_data, normalize=False, include_replacements=False, allowed=None, min_entity_count=None, include_only="entities", random_seed=18)
    assert repr((train, test)) == "([['    def __init__(self, value):\\n\\n        self.val = value\\n', {'entities': [(23, 28, 'int')]}], ['def annotated(arg1):\\n    pass\\n', {'entities': [(14, 18, 'list')]}], ['    def __init__(self, argument):\\n\\n        self.field = argument\\n', {'entities': [(23, 31, 'int')]}]], [['    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2', {'entities': [(33, 42, 'int')]}]])"

    train, test = read_data(test_data, normalize=False, include_replacements=False, allowed={"int"}, min_entity_count=None, include_only="entities", random_seed=18)
    assert repr((train, test)) == "([['    def __init__(self, value):\\n\\n        self.val = value\\n', {'entities': [(23, 28, 'int')]}], ['def annotated(arg1):\\n    pass\\n', {'entities': [(14, 18, 'Other')]}], ['    def __init__(self, argument):\\n\\n        self.field = argument\\n', {'entities': [(23, 31, 'int')]}]], [['    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2', {'entities': [(33, 42, 'int'), (64, 73, 'Other')]}]])"

    train, test = read_data(test_data, normalize=False, include_replacements=False, allowed=None, min_entity_count=2, include_only="entities", random_seed=18)
    assert repr((train, test)) == "([['    def __init__(self, value):\\n\\n        self.val = value\\n', {'entities': [(23, 28, 'int')]}], ['def annotated(arg1):\\n    pass\\n', {'entities': [(14, 18, 'Other')]}], ['    def __init__(self, argument):\\n\\n        self.field = argument\\n', {'entities': [(23, 31, 'int')]}]], [['    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2', {'entities': [(33, 42, 'int'), (64, 73, 'Other')]}]])"

    train, test = read_data(test_data, normalize=False, include_replacements=False, allowed=None, min_entity_count=2, include_only="categories", random_seed=18)
    assert repr((train, test)) == '([[\'def main() :\\n    print(instance.method1())\\n\', {\'categories\': [\'Other\']}], [\'    def __repr__(self) :\\n\\n        return f"Number({self.val})"\', {\'categories\': [\'str\']}], [\'    def method1(self) :\\n\\n        return self.method2()\\n\', {\'categories\': [\'str\']}]], [[\'    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2\', {\'categories\': [\'str\']}]])'

    train, test = read_data(test_data, normalize=False, include_replacements=False, allowed=None, min_entity_count=None, include_only="categories", random_seed=18)
    assert repr((train, test)) == '([[\'def main() :\\n    print(instance.method1())\\n\', {\'categories\': [\'None\']}], [\'    def __repr__(self) :\\n\\n        return f"Number({self.val})"\', {\'categories\': [\'str\']}], [\'    def method1(self) :\\n\\n        return self.method2()\\n\', {\'categories\': [\'str\']}]], [[\'    def method2(self) :\\n\\n        variable1 = self.field\\n        variable2 = str(variable1)\\n        return variable2\', {\'categories\': [\'str\']}]])'


