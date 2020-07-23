while read p; do
  echo $p
  wget $(awk '{print $2}') -o $(awk '{print $4}')
  echo "Waiting 10s..."
  sleep 10s
done