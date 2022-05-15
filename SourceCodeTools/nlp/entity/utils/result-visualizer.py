import matplotlib.pyplot as plt
import sys
import os
import json

results_path = sys.argv[1]
output_path = sys.argv[2]

train_losses = []
test_losses = []
train_f1 = []
test_f1 = []
legend = []

for dir in os.listdir(results_path):
    hp_path = os.path.join(results_path, dir)

    if os.path.isdir((hp_path)):

        for trial in os.listdir(hp_path):

            trial_path = os.path.join(hp_path, trial)

            if os.path.isdir(trial_path):

                metadata_path = os.path.join(trial_path, 'params.json')
                if os.path.isfile(metadata_path):

                    with open(metadata_path) as meta:
                        metadata = json.loads(meta.read().strip())

                        if metadata['test_f1'][-1] < 0.65: continue

                        train_losses.append(metadata['train_losses'])
                        train_f1.append(metadata['train_f1'])
                        test_losses.append(metadata['test_losses'])
                        test_f1.append(metadata['test_f1'])

                        legend.append(f"{dir}_{trial}")

if not os.path.isdir(output_path):
    os.mkdir(output_path)

# plt.figure(figsize=(20, 10))
for s in test_f1:
    plt.plot(s)
plt.title("Test F1")
plt.legend(legend, loc='upper left')
plt.grid()
plt.savefig(os.path.join(output_path, "test_f1.svg"))
plt.close()

for train_loss, train_f1, test_loss, test_f1, label in zip(train_losses, train_f1, test_losses, test_f1, legend):
    # plt.figure(figsize=(20, 10))
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.yscale('log')
    plt.grid()
    plt.title(f"{label} loss")
    plt.legend(["Train", "Test"], loc='upper left')
    plt.savefig(os.path.join(output_path, f"{label}_loss.svg"))
    plt.close()

    # plt.figure(figsize=(20, 10))
    plt.plot(train_f1)
    plt.plot(test_f1)
    plt.grid()
    plt.title(f"{label} f1")
    plt.legend(["Train", "test"], loc='upper left')
    plt.savefig(os.path.join(output_path, f"{label}_f1.svg"))
    plt.close()
