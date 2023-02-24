from pathlib import Path


def _cat_files(input_dir, output_file):
    for name in input_dir.iterdir():
        if name.name.startswith("."):
            continue

        if name.is_dir():
            _cat_files(name, output_file)

        if name.is_file():
            filename = name.name
            if not filename.startswith(".") and filename.endswith(".py"):
                with open(name, "r") as source:
                    output_file.write(source.read())
                    output_file.write("\n")


def cat_files(input_dir, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = open(output_path, "w")
    _cat_files(Path(input_dir), output_file)
    output_file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_file")

    args = parser.parse_args()

    cat_files(Path(args.input_dir), Path(args.output_file))
