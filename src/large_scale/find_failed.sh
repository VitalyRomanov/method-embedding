for repo in $(ls $1); do
  if [ ! -f $1/$repo/sourcetrail.log ]; then
    echo -e "$repo\tError no index"
  else
#    cat $1/$repo/sourcetrail.log | grep "Edges 0"
    find_edges=$(cat "$1/$repo/sourcetrail.log" | grep "\s0 Edges")
    if [ -n "$find_edges" ]; then
      echo -e "$repo\tError no edges"
    fi
  fi
done