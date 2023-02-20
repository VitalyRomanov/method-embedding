import argparse
from pathlib import Path

from SourceCodeTools.code.data.file_utils import unpersist


pytype_config = """
# NOTE: All relative paths are relative to the location of this file.

[pytype]

# Space-separated list of files or directories to exclude.
exclude =
    **/*_test.py
    **/test_*.py

# Space-separated list of files or directories to process.
inputs =
    {files}

# Keep going past errors to analyze as many files as possible.
keep_going = True

# Run N jobs in parallel. When 'auto' is used, this will be equivalent to the
# number of CPUs on the host system.
jobs = 4

# All pytype output goes here.
output = {package_path}/pytype

# Platform (e.g., "linux", "win32") that the target code runs on.
platform = darwin

# Paths to source code directories, separated by ':'.
# pythonpath = .

# Python version (major.minor) of the target code.
python_version = 3.7

# Always use function return type annotations. This flag is temporary and will
# be removed once this behavior is enabled by default.
always_use_return_annotations = False

# Enable parameter count checks for overriding methods. This flag is temporary
# and will be removed once this behavior is enabled by default.
overriding_parameter_count_checks = False

# Use the enum overlay for more precise enum checking. This flag is temporary
# and will be removed once this behavior is enabled by default.
use_enum_overlay = False

# Opt-in: Do not allow Any as a return type.
no_return_any = False

# Experimental: Support pyglib's @cached.property.
enable_cached_property = False

# Experimental: Infer precise return types even for invalid function calls.
precise_return = False

# Experimental: Solve unknown types to label with structural types.
protocols = False

# Experimental: Only load submodules that are explicitly imported.
strict_import = False

# Experimental: Enable exhaustive checking of function parameter types.
strict_parameter_checks = False

# Experimental: Emit errors for comparisons between incompatible primitive
# types.
strict_primitive_comparisons = False

# Space-separated list of error names to ignore.
disable =
    annotation-type-mismatch
    assert-type
    attribute-error
    bad-concrete-type
    bad-function-defaults
    bad-return-type
    bad-slots
    bad-unpacking
    bad-yield-annotation
    base-class-error
    container-type-mismatch
    duplicate-keyword-argument
    final-error
    ignored-abstractmethod
    ignored-metaclass
    ignored-type-comment
    import-error
    incomplete-match
    invalid-annotation
    invalid-directive
    invalid-function-definition
    invalid-function-type-comment
    invalid-namedtuple-arg
    invalid-super-call
    invalid-typevar
    late-directive
    match-error
    missing-parameter
    module-attr
    mro-error
    name-error
    not-callable
    not-indexable
    not-instantiable
    not-supported-yet
    not-writable
    paramspec-error
    pyi-error
    python-compiler-error
    recursion-error
    redundant-function-type-comment
    redundant-match
    reveal-type
    signature-mismatch
    typed-dict-error
    unbound-type-param
    unsupported-operands
    wrong-arg-count
    wrong-arg-types
    wrong-keyword-args

# Don't report errors.
report_errors = False

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module_path")

    args = parser.parse_args()
    module_path = Path(args.module_path)

    files_path = module_path.joinpath("files.csv")
    edges_path = module_path.joinpath("edges_with_ast.bz2")

    files = unpersist(files_path)
    edges = unpersist(edges_path)

    file_path_map = dict(zip(files["id"], files["path"]))

    annotations = edges.query("type == 'annotation_for' or type == 'returned_by'")
    files = annotations["file_id"].apply(file_path_map.get).apply(Path).unique()
    for filename in files:
        assert "test_" not in filename.name and "_test.py" not in filename.name

    file_list = " ".join(map(str, files))

    config = pytype_config.format(
        files=file_list,
        package_path=str(module_path.absolute())
    )

    with open(module_path.joinpath("pytype.conf"), "w") as sink:
        sink.write(config)


if __name__ == "__main__":
    main()
