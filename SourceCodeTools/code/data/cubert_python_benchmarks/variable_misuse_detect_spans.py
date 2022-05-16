import json
from tqdm import tqdm


def iterate_json(path):
    with open(path) as jsonl:
        for line in jsonl:
            yield json.loads(line)


class ReplacementDetector:
    def __init__(self, original, modified):
        self.original = original
        self.modified = modified

        parts = modified["info"].split("`")
        # print(parts)
        self.original_var = parts[-4]
        self.replacement_var = parts[-2]

        self.common_prefix = 0
        for i in range(min(len(self.original_var), len(self.replacement_var))):
            if self.original_var[i] != self.replacement_var[i]:
                break
            self.common_prefix += 1

    def get_spans(self):
        pos_1 = 0
        pos_2 = 0

        o = self.original["function"]
        m = self.modified["function"]

        while pos_1 < len(o) or pos_2 < len(m):
            if o[pos_1] == m[pos_1]:
                pos_1 += 1
                pos_2 += 1
            else:
                start = pos_1 - self.common_prefix
                orig_end = start + len(self.original_var)
                repl_end = start + len(self.replacement_var)
                assert o[start: orig_end] == self.original_var and m[start: repl_end] == self.replacement_var
                assert o[orig_end:] == m[repl_end:]

                return (start, orig_end), (start, repl_end)


def detect_misuse(data_path, output_path):
    original = None
    misuse = None

    with open(output_path, "w") as sink:
        for ind, record in enumerate(tqdm(iterate_json(data_path))):
            if ind % 2 == 0:
                assert record["info"].endswith("original")

            if ind % 2 == 0:
                if original is not None:
                    d = ReplacementDetector(original, misuse)
                    o_span, m_span = d.get_spans()
                    r = {
                        "info": misuse["info"],
                        "original_function": original["function"],
                        "original_span": o_span,
                        "function": misuse["function"],
                        "misuse_span": m_span,
                        "label": "Variable misuse"
                    }
                    sink.write(f"{json.dumps(r)}\n")

                original = record
            else:
                misuse = record
                o_info = original["info"].strip("original")
                assert misuse["info"][:len(o_info)] == o_info


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("output_path")

    args = parser.parse_args()

    detect_misuse(args.data_path, args.output_path)