PROJECT_DIR=$1
FILE=$2

cd $PROJECT_DIR
# https://github.com/saltudelft/libsa4py/blob/master/libsa4py/pyre.py
pyre init
pyre start
pyre query "type('$FILE')" | python -m json.tool > pyre_annotations.json
pyre kill
