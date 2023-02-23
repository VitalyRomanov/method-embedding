PROJECT_DIR=$1
FILE=$2

cd $PROJECT_DIR
source bin/activate
pip install --upgrade pip
pip install -r packages.txt
pip install pytype==2023.2.17
pytype -V 3.7 -k -o pytype --disable pyi-error,name-error $FILE
cat $repo/packages.txt | xargs pip uninstall -y