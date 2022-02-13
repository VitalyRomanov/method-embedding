while read p; do
  reponame=$(echo $p | awk -F"/" '{print $5}')
  echo $reponame, $p
  wget -O "$reponame.zip" $p
  echo "Waiting 2s..."
  sleep 2s
done