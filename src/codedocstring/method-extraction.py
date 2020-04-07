#
# Extract method names from codedocstring data
#
#

import sys
import os
import re 
from collections import Counter
import json

code_docstring_path = sys.argv[2]
format_ = sys.argv[1]

supported_formats = {'plain', 'json'}

if format_ not in supported_formats:
    print("Supported formats: ", supported_formats)
    sys.exit()

# code_path = "/Volumes/External/dev/code-translation/code-docstring-corpus/parallel-corpus/data_ps.bodies.train"
# method_path = "/Volumes/External/dev/code-translation/code-docstring-corpus/parallel-corpus/data_ps.declarations.train"
code_path = os.path.join(code_docstring_path, "parallel-corpus/data_ps.bodies.train")
method_path = os.path.join(code_docstring_path, "parallel-corpus/data_ps.declarations.train")

decls = open(method_path).readlines()

parents = Counter()
methods = Counter()

count = 0
with open(code_path) as source:
    for ind, line in enumerate(source):
        record = {}
        method_name = re.findall('def \w+\(', decls[ind])[0][4:-1]
        
        # print(method_name+"()")
        # if method_name in parents:
        #     parents[method_name] += 1
        # else:
        #     parents[method_name] = 1

        record['super'] = [method_name+"()"]
        record['sub'] = []

        for m in re.findall('\w+\(', line):

            if format_ == 'plain':
                print(f"{method_name}()\t{m[:-1]}()", )

            record['sub'].append([m[:-1]+"()"])
            if m in methods:
                methods[m] += 1
            else:
                methods[m] = 1

        if format_ == 'json':
            print(json.dumps(record))

        # count += 1
        # # if count == 100: break

# with open("parents.txt", "w") as p:
#     for item, count in parents.most_common():
#         p.write("%s\t%d\n" % (item, count))

# with open("methods.txt", "w") as met:
#     for item, count in methods.most_common():
#         met.write("%s\t%d\n" % (item, count))

# with open("united.txt", "w") as u:
#     for item, count in (parents | methods).most_common():
#         u.write("%s\t%d\n" % (item, count))