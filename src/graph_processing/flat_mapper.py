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

emit_adjacent = False

if len(sys.argv) > 1 and sys.argv[1] == "a":

    def get_cand(seq, num=0):
        for ind, elem in enumerate(seq):
            if ind == len(seq) - 1:
                break
            cand = seq[ind + 1]
            print(elem, cand)

    emit_adjacent = True


for line in sys.stdin:
    positions = line.strip().split()

    if emit_adjacent:
        get_cand(positions)
        continue

    last_w = ""
    while positions:
        if last_w != positions[0]:
            get_cand(positions)
        last_w = positions[0]
        positions.pop(0)

    
