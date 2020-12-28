import pickle 
import sys

input_ = sys.argv[1]
output_ = sys.argv[2]

embs = pickle.load(open(input_, "rb"))[-1]

with open(output_, "w") as sink:
    n_embs = len(embs)
    n_dims = len(embs[next(iter(embs.keys()))])
    sink.write(f"{n_embs} {n_dims}\n")
    for i, key in enumerate(embs):
        sink.write(f"{key} ")
        # if i == 10:
        #     break
        for ind, v in enumerate(embs[key]):
            if ind == n_dims - 1:
                sink.write(f"{v}\n")
            else:
                sink.write(f"{v} ")

        if i % 1000 == 0:
            print(f"{i}/{n_embs}", end = "\r")