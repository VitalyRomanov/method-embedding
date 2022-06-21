from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")

    args = parser.parse_args()

    logfile_path = Path(args.logfile)
    output_path = logfile_path.parent

    with open(args.logfile, "r") as logfile:
        steps = []
        buffer_pos = []
        buffer_neg = []
        model_names = []
        scores = []
        eval_model_names = []

        def parse_train(line):
            if line.startswith("Training"):
                model_names.append(line.split()[1])
                buffer_pos.append([])
                buffer_neg.append([])
                steps.append([])
            elif "pos_loss" in line:
                buffer_pos[-1].append(float(line.split("pos_loss: ")[-1]))
                steps[-1].append(float(line.split("(")[-1].split("/")[0]))
            elif "neg_loss" in line:
                buffer_neg[-1].append(float(line.split("neg_loss: ")[-1]))

        def parse_eval(line):
            if line.startswith("Evaluating"):
                eval_model_names.append(line.split()[1])
                scores.append({})
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

        eval_mode = False
        for line in logfile:
            if line.startswith("Evaluating"):
                eval_mode = True

            if eval_mode:
                parse_eval(line)
            else:
                parse_train(line)

        assert model_names == eval_model_names

    for s, v in zip(steps, buffer_pos):
        plt.plot(s, v)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.gca().set_yscale("log")
    plt.legend(model_names)
    plt.grid()
    plt.savefig(output_path.joinpath("positive_loss.svg"))
    plt.close()

    for s, v in zip(steps, buffer_neg):
        plt.plot(s, v)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.legend(model_names)
    plt.savefig(output_path.joinpath("negative_loss.svg"))
    plt.close()

    for s, v, n in zip(steps, buffer_pos, buffer_neg):
        plt.plot(s, np.array(v) + np.array(n))
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.gca().set_yscale("log")
    plt.grid()
    plt.legend(model_names)
    plt.savefig(output_path.joinpath("overall_loss.svg"))
    plt.close()

    import pandas as pd
    s = pd.DataFrame.from_records(scores)
    s["model_names"] = model_names
    s = s.set_index("model_names")
    print(s.to_string())




if __name__ == "__main__":
    main()