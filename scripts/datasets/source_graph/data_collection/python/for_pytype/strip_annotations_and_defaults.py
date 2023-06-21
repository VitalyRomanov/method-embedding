import argparse
import json
import shutil
from pathlib import Path

from SourceCodeTools.code.data.type_annotation_dataset.create_type_annotation_dataset import process_body
from SourceCodeTools.nlp import create_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("package_path")
parser.add_argument("files_path")
parser.add_argument("output_filename")


args = parser.parse_args()

package_path = Path(args.package_path)
files_path = Path(args.files_path)
output_filename = args.output_filename
packages_to_process_path = package_path.joinpath("env_site_packages.txt")


with open(packages_to_process_path, "r") as ptp:
    packages_to_process = []
    for line in ptp:
        packages_to_process.append(line.strip())


def iterate_python_files(path, to_process=None):
    for name in path.iterdir():
        if to_process is not None and name.name not in to_process:
            continue

        if name.is_dir():
            yield from iterate_python_files(name)

        if name.is_file() and not name.name.startswith(".") and name.name.endswith(".py"):
            yield name


for file_path in iterate_python_files(files_path, set(packages_to_process)):

    print(f"Removing annotations from {file_path}")

    with open(file_path, "r") as source:
        file_content = source.read()

    entry = process_body(
        create_tokenizer("spacy"), file_content, require_labels=True, remove_default=False, remove_docstring=False,
        keep_return_offsets=True
    )

    if entry is not None and (len(entry["ents"]) > 0 or len(entry["cats"]) > 0):
        shutil.move(file_path, file_path.parent.joinpath(file_path.name + f"__before_{output_filename}__" + ".original"))
        with open(file_path, "w") as sink:
            sink.write(entry["text"])

        with open(package_path.joinpath(output_filename), "a") as sink:
            entry_str = json.dumps({
                "file_path": package_path.absolute().name + "/" + str(file_path.relative_to(package_path)),
                "variable_annotations": entry["ents"],
                "return_annotations": entry["cats"],
                "default_values": entry["defaults"]
            })
            sink.write(f"{entry_str}\n")
    else:
        print(f"No annotations found")
