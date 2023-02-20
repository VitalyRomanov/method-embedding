import argparse
import json
import shutil
from pathlib import Path

from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import process_body
from SourceCodeTools.nlp import create_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("package_path")
parser.add_argument("file_path")


args = parser.parse_args()

package_path = Path(args.package_path)
file_path = Path(args.file_path)

with open(file_path, "r") as source:
    file_content = source.read()

entry = process_body(
    create_tokenizer("spacy"), file_content, require_labels=False, remove_default=True, remove_docstring=False,
    keep_return_offsets=True
)

shutil.move(file_path, file_path.parent.joinpath(file_path.name + ".original"))

with open(file_path, "w") as sink:
    sink.write(entry["text"])

with open(package_path.joinpath("type_annotations.json"), "a") as sink:
    entry_str = json.dumps({
        "file_path": package_path.absolute().name + "/" + str(file_path.relative_to(package_path)),
        "variable_annotations": entry["ents"],
        "return_annotations": entry["cats"],
        "default_values": entry["defaults"]
    })
    sink.write(f"{entry_str}\n")
