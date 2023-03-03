from SourceCodeTools.nlp import create_tokenizer


def type_is_valid(type):
    tokenize = create_tokenizer("regex", regex="[\w\.]+")
    return (
        ("\n" not in type) and
        ("(" not in type) and
        (")" not in type) and
        ("|" not in type) and
        ("*" not in type) and
        ("<" not in type) and
        (">" not in type) and
        # (" or " not in type) and
        (len(type) > 2) and
        (type.count("[") == type.count("]")) and
        (type.count("\"") % 2 == 0) and
        (type.count("\'") % 2 == 0) and
        (type != "Optional[Any]") and
        (type != "Any") and
        (type != "Optional") and
        (type != "object") and
        (not ("," in type and "[" not in type)) and
        (not ("[" not in type and len(tokenize(type)) > 1))
    )


class TypeHierarchyParser:
    def __init__(self, type_, normalize=True, tokenize_fn=None):
        self._tokenize_fn = tokenize_fn
        if self._tokenize_fn is None:
            self._tokenize_fn = create_tokenizer("spacy")
        if normalize:
            type_ = self.normalize(type_)
        self._parse_type(type_)

    def _make_node(self, start, end):
        if end is None:
            name = "".join(t.text for t in self._tokens[start:])
        else:
            name = "".join(t.text for t in self._tokens[start:end])
        return {
            "name": name,
            "children": []
        }

    def _parse_type(self, type_string):

        self._tokens = self._tokenize_fn(type_string)

        entry_point = []
        reference_stack = [entry_point]

        start = 0

        for ind, tok in enumerate(self._tokens):
            if tok.text in {"["}:
                new_record = self._make_node(start=start, end=ind)
                reference_stack[-1].append(new_record)
                reference_stack.append(new_record["children"])
                start = ind + 1
            elif tok.text in {"]"}:
                if start != ind:# or (start > 0 and self._tokens[start-1].text == ","):
                    reference_stack[-1].append(self._make_node(start=start, end=ind))
                reference_stack.pop(-1)
                start = ind + 1
            elif tok.text in {","}:
                if start != ind or (start > 0 and self._tokens[start-1].text == "["):
                    reference_stack[-1].append(self._make_node(start=start, end=ind))
                start = ind + 1
            elif ind == len(self._tokens) - 1:
                reference_stack[-1].append(self._make_node(start=start, end=None))
                reference_stack.pop(-1)

        if len(entry_point) == 0:
            raise Exception(f"Error parsing: {self._tokens}")
        self._structure = entry_point[0]
        if len(entry_point) == 1:
            pass
        elif len(entry_point) == 2:
            self._structure["end"] = entry_point[1]["name"]
        else:
            raise Exception(f"Error parsing: {self._tokens}")

    def assemble(self, structure=None, current_level=0, max_level=-1, simplify_nodes=False):

        assert max_level != 0, "Max level can be -1 or > 0"
        if max_level != -1 and current_level == max_level:
            return ""

        if structure is None:
            structure = self._structure

        name = structure["name"]
        if simplify_nodes and not (name.startswith("\"") or name.startswith("\'") or name.startswith('"""')):
            s = name.split(".")[-1]
        else:
            s = name

        n_chld = len(structure["children"])

        if max_level != -1 and current_level < max_level - 1 and n_chld > 0 or \
            max_level == - 1 and n_chld > 0:
            s += "["
            for ind, c in enumerate(structure["children"]):
                s += self.assemble(c, current_level=current_level+1, max_level=max_level, simplify_nodes=simplify_nodes)
                if ind != n_chld - 1:
                    s += ","
            s += "]"

        if "end" in structure:
            s += structure["end"]

        return s

    def __str__(self):
        return self.assemble(self._structure)

    @staticmethod
    def normalize(type):
        return type.replace(" ", "")\
            .replace("[]", "")\
            .replace("[,]", "")\
            .replace("\"", "")\
            .replace("\'", "")\
            .replace("...", "")\
            .replace(",]", "]")\
            .replace("[,", "[")
            # .replace(',",', '",')\
            # .replace(',"]', '"]')