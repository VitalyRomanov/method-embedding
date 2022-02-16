import matplotlib.pyplot as plt
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")

    args = parser.parse_args()

    with open(args.logfile, "r") as logfile:
        steps = [[]]
        buffer_pos = [[]]
        buffer_neg = [[]]
        model_names = []
        scores = [{}]
        for line in logfile:
            if "pos_loss" in line:
                buffer_pos[-1].append(float(line.split("pos_loss: ")[-1]))
                steps[-1].append(float(line.split("(")[-1].split("/")[0]))
            elif "neg_loss" in line:
                buffer_neg[-1].append(float(line.split("neg_loss: ")[-1]))
            elif "Save model" in line:
                model_names.append(line.split("/")[-2])
                scores[-1]["Model"] = model_names[-1]
            elif "Test average MRR" in line:
                scores[-1]["MRR"] = line.split(":")[-1].strip()
            elif "Test average MR" in line:
                scores[-1]["MR"] = line.split(":")[-1].strip()
            elif "Test average HITS@1:" in line:
                scores[-1]["HITS@1"] = line.split(":")[-1].strip()
            elif "Test average HITS@3:" in line:
                scores[-1]["HITS@3"] = line.split(":")[-1].strip()
            elif "Test average HITS@10:" in line:
                scores[-1]["HITS@10"] = line.split(":")[-1].strip()
                buffer_pos.append([])
                buffer_neg.append([])
                steps.append([])
                scores.append({})

    for s, v in zip(steps, buffer_pos):
        plt.plot(np.log10(s), np.log10(v))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(model_names)
    plt.savefig("positive_loss.png")
    plt.close()

    for s, v in zip(steps, buffer_neg):
        plt.plot(np.log10(s), np.log10(v))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(model_names)
    plt.savefig("negative_loss.png")
    plt.close()

    for s, v, n in zip(steps, buffer_pos, buffer_neg):
        plt.plot(np.log10(s), np.log10(np.array(v) + np.array(n)))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(model_names)
    plt.savefig("overall_loss.png")
    plt.close()

    import pandas as pd
    s = pd.DataFrame.from_records(scores)
    print(s.to_string)




if __name__ == "__main__":
    main()