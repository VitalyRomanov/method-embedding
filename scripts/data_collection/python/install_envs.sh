conda activate python37

while read repo
do
#  echo $repo/bin/python
  # Create environment if does not exist
  if [ ! -f "$repo/bin/python" ]
  then
    echo "Creating environment for $repo"
    if [ -z "$(python -m venv $repo)" ]
    then
      echo "Created environment for $repo"
      cr_env=true
    else
      cr_env=false
    fi
  else
    echo "Environment for $repo exists"
    cr_env=true
  fi

  # Activate environment if created
  if $cr_env; then
    source $repo/bin/activate && act_env=true
    echo "Activated $repo"
  fi

  # Process
  if $act_env; then
    # install packages
    if [ ! -f "$repo/packages.txt" ]; then
      echo "Installing packages for $repo"
      pip install $repo > $repo/piplog.log
      pip freeze > $repo/packages.txt
#    else
#      cat $repo/packages.txt | xargs pip install > $repo/piplogsecondary.log
    fi

    if [ ! -f "$repo/$repo.srctrlprj" ]; then
      echo "Creating Sourcetrail project for $repo"
      repo_lower=$(echo "$repo" | awk '{print tolower($0)}')
      repo_lower_under="${repo_lower//[-]/_}"
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
                <source_path>lib/python3.7/site-packages/$repo_lower_under</source_path>
            </source_paths>
            <status>enabled</status>
            <type>Python Source Group</type>
        </source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
    </source_groups>
    <version>8</version>
</config>" > $repo/$repo.srctrlprj
    fi

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
      # try to install packages (maybe environemnt is empty)
      cat $repo/packages.txt | xargs pip install > $repo/piplogsecondary.log
      Sourcetrail.sh index -i $repo/$repo.srctrlprj >> $repo/sourcetrail.log
      cat $repo/packages.txt | xargs pip uninstall -y > $repo/pipuninstlog.log
    else
      echo "Already indexed"
    fi
  fi

  # Deactivate environment if activated
  if $act_env; then
    deactivate
    echo "Deactivated $repo"
  fi

#  # Activate environment
#  if [ -z "$cr_env" ]
#  then
#    source $repo/bin/activate
#    act_env=""
#
#    if [ -z "$act_env" ]
#    then
#      echo "Activated environment"
#      echo "$(which pip)"
#
#      if [[ -f "$repo/sourcetrail.log" ]]; then
#
#          echo "$FILE exist"
#      fi
#      pip install $repo > $repo/piplog.log
#      pip freeze > $repo/packages.txt
#      repo_lower=$(echo "$repo" | awk '{print tolower($0)}')
#      echo "<?xml version="1.0" encoding="utf-8" ?>
#<config>
#    <source_groups>
#        <source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
#            <name>Python Source Group</name>
#            <python_environment_path>.</python_environment_path>
#            <source_extensions>
#                <source_extension>.py</source_extension>
#            </source_extensions>
#            <source_paths>
#                <source_path>lib/python3.7/site-packages/$repo_lower</source_path>
#            </source_paths>
#            <status>enabled</status>
#            <type>Python Source Group</type>
#        </source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
#    </source_groups>
#    <version>8</version>
#</config>" > $repo/$repo.srctrlprj
#      Sourcetrail.sh index -f $repo/$repo.srctrlprj > $repo/sourcetrail.log
#      deactivate
#    else
#      echo "Problems activating, skipping"
#    fi
#  else
#    echo "Problems creating, skipping"
#  fi

done < "${1:-/dev/stdin}"

conda deactivate
