import sys

all_el = []

for line in sys.stdin:
    if line.strip():
        l = line.strip().split("/")
        all_el.append("/".join(l[-2:]))

aset = set(all_el)
# print("Total: ", len(all_el))
# print("Unique: ", len(aset))

from pprint import pprint

# https://codeload.github.com/gousiosg/java-callgraph/zip/master

for a in aset:
    parts = a.split("/")
    link = "https://codeload.github.com/%s/%s/zip/master" % (parts[-2], parts[-1].split(".")[0])
    print("wget %s -O %s.zip" % (link, parts[-1].split(".")[0]))
    print("sleep 0.5s")

# for a in aset:
#     print(a)