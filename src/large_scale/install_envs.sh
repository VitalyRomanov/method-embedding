conda activate python37

while read repo
do
  cr_env=$(python -m venv $repo)
  if [ -z "$cr_env" ]
  then
    echo "Created environment"
    source $repo/bin/activate
    act_env=""
    if [ -z "$act_env" ]
    then
      echo "Activated environment"
      echo "$(which pip)"
      pip install $repo > $repo/piplog.log
      pip freeze > $repo/packages.txt
      repo_lower=$(echo "$repo" | awk '{print tolower($0)}')
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
                <source_path>lib/python3.7/site-packages/$repo_lower</source_path>
            </source_paths>
            <status>enabled</status>
            <type>Python Source Group</type>
        </source_group_c140f8c4-74c6-42e2-bc72-982c3f658806>
    </source_groups>
    <version>8</version>
</config>" > $repo/$repo.srctrlprj
      Sourcetrail.sh index -f $repo/$repo.srctrlprj > $repo/sourcetrail.log
      deactivate
    else
      echo "Problems activating, skipping"
    fi
  else
    echo "Problems creating, skipping"
  fi

done < "${1:-/dev/stdin}"

conda deactivate