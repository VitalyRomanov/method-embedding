import sys
import json
from collections import Counter

all_ents = []
all_cls = []

def preprocess(ent):
    return ent.strip("\"").split("[")[0].split(".")[-1]

with open(sys.argv[1], "r") as data:
    for line in data:
        entry = json.loads(line)
        try:
            ents = [preprocess(e[2]) for e in entry['ents']]
            all_ents.extend(ents)
        except:
            pass
        try:
            all_cls.extend(preprocess(e['returns']) for e in entry['cats'])
        except:
            pass

print("Unique classes")
for class_, count in Counter(all_cls).most_common():
    print(f"{class_}\t{count}")

print("\n\n\nUnique ents")
for ent, count in Counter(all_ents).most_common():
    print(f"{ent}\t{count}")

with open("allowed.txt", "w") as sink:
    print("\n\n\nOverall ents")
    for ent, count in Counter(all_ents+all_cls).most_common():
        print(f"{ent}\t{count}")
        if count > 50:
            sink.write((f"{ent}\n"))

