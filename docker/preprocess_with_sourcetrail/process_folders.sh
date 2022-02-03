#!/bin/bash

# this script processes arbitrary code with sourcetrails, no dependencies are pulled
PYTHON_ENV=$(realpath $1)
DATA_PATH=$2
#RUN_DIR=$(realpath "$(dirname "$0")")
RUN_DIR=$APP_PATH
VERIFIER_PATH="$RUN_DIR/sourcetrail_verify_files.py"

create_sourcetrail_project_if_not_exist () {
  PYTHON_ENV=$1
  PRJ_PATH=$2
   if [ ! -f $PRJ_PATH ]; then
    echo "Creating Sourcetrail project for $repo"
    echo "<?xml version="1.0" encoding="utf-8" ?>
<config>
  <source_groups>
      <source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
          <name>Python Source Group</name>
          <python_environment_path>$PYTHON_ENV</python_environment_path>
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
</config>" > $PRJ_PATH
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

#echo "Running from $RUN_DIR"
#echo "Python env path $PYTHON_ENV"
#echo "Data path $DATA_PATH"
#echo "Verifier path $VERIFIER_PATH"

for repo_dir in $DATA_PATH/*
do
#  echo $repo_dir
#  repo=$DATA_PATH/$folder
  if [ -d "$repo_dir" ]
  then
    repo="$(basename $repo_dir)"
    echo $repo
    cd $DATA_PATH
    create_sourcetrail_project_if_not_exist $PYTHON_ENV "$repo/$repo.srctrlprj"
    run_indexer "$repo"

    if [ -f "$repo/$repo.srctrldb" ]; then
      cd "$repo"
      echo $(pwd)
      echo ".headers on
.mode csv
.output edges.csv
SELECT * FROM edge;
.output nodes.csv
SELECT * FROM node;
.output element_component.csv
SELECT * FROM element_component;
.output source_location.csv
SELECT * FROM source_location;
.output occurrence.csv
SELECT * FROM occurrence;
.output filecontent.csv
SELECT * FROM filecontent;
.quit" | sqlite3 "$repo.srctrldb"
      python $VERIFIER_PATH .
      cd ..
    else
      echo "Package not indexed"
    fi

    cd $RUN_DIR
  fi
done
