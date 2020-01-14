import sys

all_el = []

for line in sys.stdin:
    if line.strip():
        l = line.strip().split(" ")[0].split("/")
        l[0] = "https://github.com"
        l[2] = l[2] + ".git"
        all_el.append("/".join(l[:3]))
        # print(all_el)

aset = set(all_el)
# print("Total: ", len(all_el))
# print("Unique: ", len(aset))

from pprint import pprint

# https://codeload.github.com/gousiosg/java-callgraph/zip/master

for a in aset:
    parts = a.split("/")
    link = "https://codeload.github.com/%s/%s/zip/master" % (parts[3], parts[4].split(".")[0])
    print("wget %s -o %s.zip" % (link, parts[4].split(".")[0]))
    print("sleep 5s")

# for a in aset:
#     print(a)