if [ -d code ]; then 
    rm -rf code
    rm -rf extracted
fi

mkdir code
mkdir extracted

while read line
do
    repo_name=$(echo $line | awk -F"\t" '{print $2}')
    repo_link=$(echo $line | awk -F"\t" '{print $5}')
    
    echo "$repo_name"
    echo "$repo_link"
    
    wget $repo_link
    unzip master -d extracted
    
    find . -name "*.py" | python merge
    
    # for file in $(find . -name "*.py"); do
    #     cat $file >> code/codes.txt
    # done
    
    sh -c "rm -rf extracted/*"
    rm -rf master
    
    sleep 0.5

done < "${1:-/dev/stdin}"

