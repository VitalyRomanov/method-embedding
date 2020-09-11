import pandas as pd
import ast# , astor
import sys, os
import json
import spacy
import re
from SourceCodeTools.proc.entity.util import inject_tokenizer
from ast import literal_eval

from spacy.gold import biluo_tags_from_offsets

nlp = inject_tokenizer(spacy.blank("en"))

allowed = {'str','bool','Optional','None','int','Any','Union','List','Dict','Callable','ndarray','FrameOrSeries','bytes','DataFrame','Matcher','float','Tuple','bool_t','Description','Type'}

def preprocess(ent):
    return ent
    # return ent.strip("\"").split("[")[0].split(".")[-1]

def inspect_fdef(node):
    if node.returns is not None:
        return [{"name": "returns", "line": node.returns.lineno, "col_offset": node.returns.col_offset, "end_col_offset": node.returns.end_col_offset}]
    else:
        return []

def inspect_arg(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset}]
    else:
        return []

def inspect_ann(node):
    if node.annotation is not None:
        return [{"name": "annotation", "line": node.annotation.lineno, "col_offset": node.annotation.col_offset, "end_col_offset": node.annotation.end_col_offset, "var_col_offset": node.col_offset, "var_end_col_offset": node.end_col_offset}]
    else:
        return []

def get_cum_lens(body):
    body_lines = body.split("\n")
    cum_lens = [0]
    for ind, line in enumerate(body_lines):
        cum_lens.append(len(line) + cum_lens[-1] + 1)
    return cum_lens


def get_initial_labels(body_):
    try:
        root = ast.parse(body_)
    except:
        return None

    positions = []
    for node in ast.walk(root):
        if isinstance(node, ast.FunctionDef):
            positions.extend(inspect_fdef(node))
        elif isinstance(node, ast.arg):
            positions.extend(inspect_arg(node))
        elif isinstance(node, ast.AnnAssign):
            positions.extend(inspect_ann(node))

    if positions:
        df = pd.DataFrame(positions)
        df.sort_values(by=['line', 'end_col_offset'], ascending=[True, False], inplace=True)

        return df
    else:
        return None

def strip_docstring(body): # remove first docstring (docstring of the main function (not nested functions)
    # # TODO
    # # this does not seem to be working well
    # root = ast.parse(body)
    # main_doc = None
    # ranges = []
    # for node in ast.walk(root):
    #     try:
    #         docstring = ast.get_docstring(node)
    #     except:
    #         continue
    #     else:
    #         if docstring is not None:
    #             ranges.append((node.body[0].lineno-1, node.body[0].end_lineno))
    #             # if main_doc is None:
    #             #     main_doc = docstring
    #             # node.body = node.body[1:]
    #
    # ranges = sorted(ranges, key=lambda x: x[0], reverse=True)

    body_lines = body.split("\n")

    new_body = []
    new_doc = []

    block = False
    no_removals = False

    docstring_starts = 0
    doc_len_lines = 0

    block_symbol = ""

    for ind, line in enumerate(body_lines):
        if not block and line.lstrip().startswith('"""') or line.lstrip().startswith("'''"):
            block = True
            block_symbol = line.lstrip()[:3]
            # continue

        if block is False or no_removals is True:
            new_body.append(line)
        else:
            new_doc.append(line)
            docstring_starts = ind
            # doc_len_lines = 1

        if block and line.rstrip().endswith(block_symbol) or line.rstrip().endswith(block_symbol):
            if len(new_doc) == 1 and len(line.strip()) >= 6 or \
                len(new_doc) > 1:
                block = False
                no_removals = True
                block_symbol = ""
                # continue

    doc_len_lines += len(new_doc)

    new_body = "\n".join(new_body)
    new_doc = "\n".join(new_doc)

    return new_body, new_doc, docstring_starts, doc_len_lines
    # return astor.to_source(root), main_doc

def isvalid(text, ents):
    doc = nlp(text)
    tags = biluo_tags_from_offsets(doc, ents)
    for t, tag in zip(doc, tags):
        print(tag, t.text, sep="\t")
    if "-" in tags:
        return False
    else:
        return True


def overlap(p1, p2):
    if (p2[1] - p1[0]) * (p2[0] - p1[1]) <= 0:
        return True
    else:
        return False

def label_replacements_overap(labels, replacements):
    evict = set()
    for ind, row in labels.iterrows():
        for ind, r in enumerate(replacements):
            if r[0] == row.line - 1:
                if overlap((row.col_offset, row.end_col_offset), (r[1], r[2])):
                    evict.add(ind)

    return [r for ind, r in enumerate(replacements) if ind not in evict]

def adjust_replacements(replacements, doc_start, doc_len):
    repl = []

    for r in replacements:
        line = r[0]
        if line > doc_start:
            line -= doc_len

        repl.append((line, r[1], r[2], r[3]))

    return repl

def account_for_strip(replacements, first_line_strip):
    repl = []
    for r in replacements:
        line = r[0]
        col_s = r[1]
        col_e = r[2]

        if line == 0:
            col_s -= first_line_strip
            col_e -= first_line_strip

        repl.append((line, col_s, col_e, r[3]))

    return repl


def prepare_replacements(replacements):
    return pd.DataFrame(
        [{"line": r[0], "col_offset": r[1], "end_col_offset": r[2], "ann": r[3]} for r in replacements]).sort_values(
        by=['line', 'end_col_offset'], ascending=[True, False])


def adjust_contraction(entry, line, head, contraction):
    for i in range(len(entry["ents"])):
        tline, start, end, ann = entry["ents"][i]
        if tline == line:
            entry["ents"][i] = (tline, start - contraction, end - contraction, ann)

    for i in range(len(entry['replacements'])):
        tline, start, end, ann = entry["replacements"][i]
        if tline == line:
            if start >= len(head):
                entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)

    return entry

def process_body(body, replacements, remove_docstring=True):
    body_ = body.strip()

    replacements = account_for_strip(replacements, first_line_strip=len(body)-len(body.lstrip()))

    entry = {"ents": [], "cats": [], "replacements": []}

    if remove_docstring:
        try:
            only_body, doc, doc_start, doc_len = strip_docstring(body_)
            o_r = replacements
            replacements = adjust_replacements(replacements, doc_start, doc_len)
        except UnicodeDecodeError:
            return None
        except SyntaxError:
            return None

        entry['text'] = only_body
        entry['docstring'] = doc

        body_ = only_body

    initial_labels = get_initial_labels(body_)

    # wrap everything in dataframe and sort
    # replacements = prepare_replacements(replacements)


    if initial_labels is not None:

        body_lines = body_.split("\n")

        replacements = label_replacements_overap(initial_labels, replacements)

        entry['replacements'].extend(replacements)

        # initial labels are sorted
        #
        for ind, row in initial_labels.iterrows():
            line = row.line - 1

            # TODO
            # multiline annotations are not parsed correctly

            annotation = body_lines[line][row.col_offset: row.end_col_offset]
            annotation = preprocess(annotation)
            tail = body_lines[line][row.end_col_offset:]
            head = body_lines[line][:row.col_offset].rstrip()
            before_contraction = len(body_lines[line])

            if row['name'] == "returns":
                try:
                    assert head.endswith("->")
                except AssertionError:
                    return None
                head = head[:-2]
                if line == 0: # only use labels for the main and not nested functions
                    entry["cats"].append({"returns": annotation})

                contraction = before_contraction - len(head) - len(tail)

                entry = adjust_contraction(entry, line, head, contraction)

                # for i in range(len(entry["ents"])):
                #     tline, start, end, ann = entry["ents"][i]
                #     if tline == line:
                #         entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
                #
                # for i in range(len(entry['replacements'])):
                #     tline, start, end, ann = entry["replacements"][i]
                #     if tline == line:
                #         if start >= len(head):
                #             entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)

            elif row['name'] == "annotation":

                try:
                    assert head.endswith(':')
                except AssertionError:
                    return None
                head = head[:-1]
                contraction = before_contraction - len(head) - len(tail)

                entry = adjust_contraction(entry, line, head, contraction)

                # for i in range(len(entry["ents"])):
                #     tline, start, end, ann = entry["ents"][i]
                #     if tline == line:
                #         entry["ents"][i] = (tline, start - contraction, end - contraction, ann)
                #
                # for i in range(len(entry['replacements'])):
                #     tline, start, end, ann = entry["replacements"][i]
                #     if tline == line:
                #         if start >= len(head):
                #             entry["replacements"][i] = (tline, start - contraction, end - contraction, ann)


                assert int(row.var_col_offset) != len(head)

                entry["ents"].append((line, int(row.var_col_offset), len(head), annotation))
            else:
                raise Exception("wtf")

            new_line = head + tail
            body_lines[line] = new_line

        entry['text'] = "\n".join(body_lines)

        cum_lens = get_cum_lens(entry['text'])

        ents_before =  [body_lines[line][start: end] for
                         ind, (line, start, end, annotation) in enumerate(entry["ents"])]

        entry["ents"] = [(cum_lens[line] + start, cum_lens[line] + end, annotation) for
                         ind, (line, start, end, annotation) in enumerate(entry["ents"])]

        entry["replacements"] = [(cum_lens[line] + start, cum_lens[line] + end, annotation) for
                         ind, (line, start, end, annotation) in enumerate(entry["replacements"])]

        ents_after = [entry['text'][start: end] for
                       ind, (start, end, annotation) in enumerate(entry["ents"])]

        assert ents_before == ents_after

        entry['original'] = body

        # entry["ents"] = list(map(lambda x: x if x[2] in allowed else (x[0], x[1], "Other"), entry["ents"]))
        # entry["cats"] = list(filter(lambda x: x["returns"] if x["returns"] in allowed else "Other", entry['cats']))

        if not entry["ents"]: # and not entry["cats"]:
            return None # in case all entities were filtered

        # assert isvalid(entry['text'], entry["ents"])
        assert isvalid(entry['text'], entry["replacements"])
        return entry

        # if isvalid(entry['text'], entry["ents"]):
        #     return entry
        # else:
        #     return None
    else:
        return None

def to_global_ids(entry, id_map, local_names, global_names):

    replacements = []
    for r in entry['replacements']:
        id_ = int(r[2].split("_")[-1])
        assert local_names[id_] == global_names[id_map[id_]], f"{local_names[id_]} != {global_names[id_map[id_]]}"
        # assert local_names[id_][0].lower() == local_names[id_][0], f"{local_names[id_]}"
        replacements.append((r[0], r[1], str(id_map[id_])))

    entry['replacements'] = replacements
    return entry

def main(args):
    bodies_path = args[1]
    bodies = pd.read_csv(bodies_path)
    id2global = pd.read_csv(args[3])
    id_maps = dict(zip(id2global['id'], id2global['global_id']))

    global_names = pd.read_csv(args[4])
    local_names = pd.read_csv(args[5])
    global_names = dict(zip(global_names['id'].tolist(), global_names['serialized_name'].tolist()))
    local_names = dict(zip(local_names['id'].tolist(), local_names['serialized_name'].tolist()))

    # body_field = bodies.columns[1]

    data = []

    for ind, (_, row) in enumerate(bodies.iterrows()):
        # b = """def cosine(w: float, A: float = 1, phi: float = 0, offset: float = 0) -> \"partial[Callable[[], None]]\":\n    ''' Return a driver function that can advance a sequence of cosine values.\n\n    .. code-block:: none\n\n        value = A * cos(w*i + phi) + offset\n\n    Args:\n        w (float) : a frequency for the cosine driver\n        A (float) : an amplitude for the cosine driver\n        phi (float) : a phase offset to start the cosine driver with\n        offset (float) : a global offset to add to the driver values\n\n    '''\n    from math import cos\n    def f(i: float) -> float:\n        return A * cos(w*i + phi) + offset\n    return partial(force, sequence=_advance(f))"""
        # b = "def _detect(executable) :\n    return shutil.which(executable)"
        # b = "def _bundle_extensions(objs, resources: Resources) :\n    names = set()\n    bundles = []\n\n    extensions = [\".min.js\", \".js\"] if resources.minified else [\".js\"]\n\n    for obj in _all_objs(objs) if objs is not None else Model.model_class_reverse_map.values():\n        if hasattr(obj, \"__implementation__\"):\n            continue\n        name = obj.__view_module__.split(\".\")[0]\n        if name == \"bokeh\":\n            continue\n        if name in names:\n            continue\n        names.add(name)\n        module = __import__(name)\n        this_file = abspath(module.__file__)\n        base_dir = dirname(this_file)\n        dist_dir = join(base_dir, \"dist\")\n\n        ext_path = join(base_dir, \"bokeh.ext.json\")\n        if not exists(ext_path):\n            continue\n\n        server_prefix = f\"{resources.root_url}static/extensions\"\n        package_path = join(base_dir, \"package.json\")\n\n        pkg = None\n\n        if exists(package_path):\n            with open(package_path) as io:\n                try:\n                    pkg = json.load(io)\n                except json.decoder.JSONDecodeError:\n                    pass\n\n        artifact_path\n        server_url\n        cdn_url = None\n\n        if pkg is not None:\n            pkg_name = pkg[\"name\"]\n            pkg_version = pkg.get(\"version\", \"latest\")\n            pkg_main = pkg.get(\"module\", pkg.get(\"main\", None))\n            if pkg_main is not None:\n                cdn_url = f\"{_default_cdn_host}/{pkg_name}@^{pkg_version}/{pkg_main}\"\n            else:\n                pkg_main = join(dist_dir, f\"{name}.js\")\n            artifact_path = join(base_dir, normpath(pkg_main))\n            artifacts_dir = dirname(artifact_path)\n            artifact_name = basename(artifact_path)\n            server_path = f\"{name}/{artifact_name}\"\n        else:\n            for ext in extensions:\n                artifact_path = join(dist_dir, f\"{name}{ext}\")\n                artifacts_dir = dist_dir\n                server_path = f\"{name}/{name}{ext}\"\n                if exists(artifact_path):\n                    break\n            else:\n                raise ValueError(f\"can't resolve artifact path for '{name}' extension\")\n\n        extension_dirs[name] = artifacts_dir\n        server_url = f\"{server_prefix}/{server_path}\"\n        embed = ExtensionEmbed(artifact_path, server_url, cdn_url)\n        bundles.append(embed)\n\n    return bundles"
        # entry = process_body(b)
        body = row['body']
        replacements = literal_eval(row['replacement_list'])
        entry = process_body(body, replacements)
        if entry is not None:
            entry = to_global_ids(entry, id_maps, local_names, global_names)
            data.append(entry)

    format = "jsonl"
    if format == "jsonl":
        with open(args[2], "a") as sink:
            for entry in data:
                sink.write(f"{json.dumps(entry)}\n")
    elif format == "csv":
        if os.path.isfile(args[2]):
            header = False
        else:
            header = True
        pd.DataFrame(data).to_csv(args[2], index=False, header=header)



if __name__ == "__main__":
    main(sys.argv)