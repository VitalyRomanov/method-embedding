import os
import sys

doc_string_path = sys.argv[1]
output = sys.argv[2]

lim = 10000

with open(doc_string_path) as bodies:
    for ind, body in enumerate(bodies):
        temp = body.strip()
        temp = temp.replace("DCNL  ", "\n")
        temp = temp.replace("DCNL ", "\n")
        temp = temp.replace("DCSP", "")

        fld = os.path.join(output, repr(ind))
        os.mkdir(fld)
        with open(os.path.join(fld, "main.py"), "w") as sink:
            sink.write(temp)
            sink.write("\n")
        if ind % 1000 == 0: print("%d\r" % ind, end='')
        if ind == lim :
            break