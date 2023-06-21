SEARCH_PATH=$1

if [ ! -f "$repo/type_annotations.json" ]; then
  python strip_annotations_and_defaults.py $SEARCH_PATH .
#  for file in $(find $SEARCH_PATH -name "*.py"); do
#    python strip_annotations_and_defaults.py $SEARCH_PATH $file
#  done
fi