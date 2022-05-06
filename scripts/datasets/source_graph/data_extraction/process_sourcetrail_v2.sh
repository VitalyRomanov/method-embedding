conda activate SourceCodeTools

ENVS_DIR=$(realpath "$1")
RUN_DIR=$(realpath "$(dirname "$0")")
SQL_Q=$(realpath "$RUN_DIR/extract.sql")

for dir in "$ENVS_DIR"/*; do
  if [ -d "$dir" ]; then
    package_name="$(basename "$dir")"

    echo "Found package $package_name"

    if [ -f "$dir/$package_name.srctrldb" ]; then
      cd "$dir"
      sqlite3 "$dir/$package_name.srctrldb" < "$SQL_Q"
      cd "$RUN_DIR"
      sourcetrail_verify_files.py "$dir"
    else
      echo "Package not indexed"
    fi
  fi
done

conda deactivate