# this script processes arbitrary code with sourcetrails, no dependencies are pulled

conda activate SourceCodeTools

create_sourcetrail_project_if_not_exist () {
   if [ ! -f $1 ]; then
    echo "Creating Sourcetrail project for $repo"
    echo "<?xml version="1.0" encoding="utf-8" ?>
<config>
  <source_groups>
      <source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
          <name>Python Source Group</name>
          <python_environment_path>.</python_environment_path>
          <source_extensions>
              <source_extension>.py</source_extension>
          </source_extensions>
          <source_paths>
              <source_path>.</source_path>
          </source_paths>
          <status>enabled</status>
          <type>Python Source Group</type>
      </source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
  </source_groups>
  <version>8</version>
</config>" > $1
  fi
}


run_indexer () {
  repo=$1
  if [ ! -f "$repo/sourcetrail.log" ]; then
    run_indexing=true
  else
    find_edges=$(cat "$repo/sourcetrail.log" | grep " Edges")
    if [ -z "$find_edges" ]; then
      echo "Indexing was interrupted, recovering..."
      run_indexing=true
    else
      run_indexing=false
    fi
  fi

  if $run_indexing; then
    echo "Begin indexing"
    Sourcetrail.sh index -i $repo/$repo.srctrlprj >> $repo/sourcetrail.log
  else
    echo "Already indexed"
  fi
}


while read repo
do
  create_sourcetrail_project_if_not_exist "$repo/$repo.srctrlprj"
  run_indexer "$repo"
done < "${1:-/dev/stdin}"

conda deactivate
