import json
import pandas as pd
import sys
import matplotlib.pyplot as plt

with open(sys.argv[1]) as res:
    data = json.loads(res.read())

overall = pd.DataFrame([{key: e[key] for key in ['ents_p', 'ents_r', 'ents_f']} for e in data])
per_type = [] #pd.DataFrame([e['ents_per_type'] for e in data])
for epoch, e in enumerate(data):
    types = list(e['ents_per_type'].keys())
    entries = []
    for type_ in types:
        entries.append({"epoch": epoch, "type": type_, "p": e['ents_per_type'][type_]['p'], "r": e['ents_per_type'][type_]['r'], "f": e['ents_per_type'][type_]['f']})
    per_type.extend(entries)
per_type = pd.DataFrame(per_type)

plt.bar(overall.index, overall['ents_f'])
plt.xlabel("Epoch")
plt.ylabel("f-score")
plt.title("Overall entity recognition score")
plt.show()

d = per_type.query("epoch == 89")
plt.bar(d['type'], d['f'])
plt.title("Entity recognition per type")
plt.ylabel("f-score")
plt.xticks(fontsize=7, rotation=70)
plt.show()
