import sys

def get_cand(seq, num=4):

    last_cand = ""
    
    if len(seq) > 1:
        for ind, cand in enumerate(seq[1:]):
            if ind == num:
                break
            if last_cand == cand: continue
            print(seq[0], cand)
            last_cand = cand

    


for line in sys.stdin:
    positions = line.strip().split()

    last_w = ""
    while positions:
        if last_w != positions[0]:
            get_cand(positions)
        last_w = positions[0]
        positions.pop(0)

    
