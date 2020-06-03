for repo in $(awk -F"\t" '{print $2}')
do
  var=$(conda search "$repo" | grep "^$repo .*py37")
#  var=$(pip search "$repo" | grep "^$repo ")
#  echo $var
  if [ -n "$var" ];
  then
    echo $repo
  fi
done < "${1:-/dev/stdin}"