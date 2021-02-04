import sys
import os

# find . -name "*.py" | python merge.py

with open("code/codes.txt", "a") as out:
    for line in sys.stdin:
        fname = line.strip()
        if os.path.isfile(fname):
            try:
                with open(line.strip(), encoding="utf-8") as data:
                    out.write(f"{data.read()}\n")
            except:
                with open(line.strip(), encoding="ISO-8859-1") as data:
                    out.write(f"{data.read()}\n")
    
    
