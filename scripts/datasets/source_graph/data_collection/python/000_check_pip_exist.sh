conda activate python37

#for repo in $(awk -F"\t" '{print $2}')
for repo in $(awk -F"\t" '{print $0}')
do
#  var=$(conda search "$repo" | grep "^$repo .*py37")
  var=$(pip search "$repo" | grep "^$repo ")
#  echo $var
  if [ -n "$var" ];
  then
    echo $repo
  fi
done < "${1:-/dev/stdin}"

conda deactivate