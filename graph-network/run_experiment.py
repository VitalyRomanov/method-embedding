import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Experiments import Experiments, Experiment
import argparse

from classifiers import LRClassifier, NNClassifier, ElementPredictor, NodeClassifier
import tensorflow as tf; tf.get_logger().setLevel('ERROR')
import numpy as np
from copy import deepcopy

parser = argparse.ArgumentParser(description="""
apicall creates an experiment where we try to predict the exstense of "next call" link between nodes. 
    Currently, the process is trained using negative sampling procedure and additional classifier. Negative sampling 
    is done uniformly form the collection of all the rest of the nodes. The number of negative samples is the batch is 
    equal to the number of positive samples. The embeddigns themselves are not trained, only weights of classifier are 
    updated. 
""")
# parser.add_argument('experiment', default=None,
#                     help='Select experiment [apicall|link|typeuse|varuse|fname|nodetype]')
parser.add_argument("--base_path", default=None, help="path to the trained GNN model")
parser.add_argument('--random', action='store_true')
args = parser.parse_args()

# GAT
# BASE_PATH = "models/GAT-2020-05-05-17-23-39-269036-fname" # trained on function names
# BASE_PATH = "models/GAT-2020-05-05-01-25-38-208819-varnames" # trained on variable names
# BASE_PATH = "models/GAT-2020-05-04-01-25-52-792623-nextcall" # trained on next call
# BASE_PATH = "models/GAT-2020-05-10-23-27-05-982293-multitask"

# RGCN
# BASE_PATH = "models/RGCN-2020-05-09-16-43-46-984454-fname-edgetype" # trained on function names
# BASE_PATH = "models/RGCN-2020-05-08-21-35-13-542497-varname-edgetype" # trained on variable names
# BASE_PATH = "models/RGCN-2020-05-06-19-53-15-933048-nextcall-edgetypes" # trained on next call
# BASE_PATH = "models/RGCN-2020-05-11-10-14-50-783337-multitask"
BASE_PATH = "models/RGCN-2020-05-11-20-34-49-002750-multitask-5layers"

# data files
API_SEQ = "data_files/python_flat_calls.csv.bz2"
VAR_USE = "data_files/python_node_to_var.csv.bz2"

e = Experiments(base_path=BASE_PATH,
                api_seq_path=API_SEQ,
                type_use_path=None, #not needed
                node_type_path=None, #not needed
                variable_use_path=VAR_USE, #not needed
                function_name_path=None,
                gnn_layer=-1
                )

# if args.random:
#     e.embed.e = np.random.randn(e.embed.e.shape[0], e.embed.e.shape[1])

# EXPERIMENT_NAME = args.experiment

def run_experiment(EXPERIMENT_NAME, random=False):
    experiment = e[EXPERIMENT_NAME]

    ma_train = 0.
    ma_test = 0.
    ma_alpha = 2 / (10 + 1)

    if random:
        experiment.embed.e = deepcopy(experiment.embed.e)
        experiment.embed.e = np.random.randn(experiment.embed.e.shape[0], experiment.embed.e.shape[1])

    if EXPERIMENT_NAME in {'link', 'apicall', 'typeuse'}:
        clf = NNClassifier(experiment.embed_size)
    elif EXPERIMENT_NAME in {'varuse', 'fname'}:
        clf = ElementPredictor(experiment.embed_size, experiment.unique_elements, 100)
    elif EXPERIMENT_NAME in {'nodetype'}:
        clf = NodeClassifier(experiment.embed_size, experiment.unique_elements)
    else:
        raise ValueError(f"Unknown experiment: {type}. The following experiments are available: [apicall|link|typeuse|varuse|fname|nodetype].")

    # clf.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = clf(**batch, training=True)
            loss = loss_object(batch["y"], predictions)
            gradients = tape.gradient(loss, clf.trainable_variables)
            optimizer.apply_gradients(zip(gradients, clf.trainable_variables))

        train_loss(loss)
        train_accuracy(batch["y"], predictions)

    @tf.function
    def test_step(batch):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = clf(**batch, training=False)
        t_loss = loss_object(batch["y"], predictions)

        test_loss(t_loss)
        test_accuracy(batch["y"], predictions)

    EPOCHS = 500

    # print(f"\n\n\nExperiment name: {EXPERIMENT_NAME}")
    tests = []

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()


        for batch_ind, batch in enumerate(experiment.train_batches()):
            train_step(batch)

        if epoch % 1 == 0:

            for batch in experiment.test_batches():
                test_step(batch)

            ma_train = train_accuracy.result() * 100 * ma_alpha + ma_train * (1 - ma_alpha)
            ma_test = test_accuracy.result() * 100 * ma_alpha + ma_test * (1 - ma_alpha)
            tests.append(ma_test)

            # template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Average Test {:.4f}'
            # print(template.format(epoch+1,
            #                       train_loss.result(),
            #                       train_accuracy.result()*100,
            #                       test_loss.result(),
            #                       test_accuracy.result()*100,
            #                       ma_test))

    # ma_train = train_accuracy.result() * 100 * ma_alpha + ma_train * (1 - ma_alpha)
    # ma_test = test_accuracy.result() * 100 * ma_alpha + ma_test * (1 - ma_alpha)

    return ma_train, max(tests)

for experiment_name in ['apicall','link','typeuse','varuse','fname','nodetype']:
    print(f"\n{experiment_name}:")
    train_acc, test_acc = run_experiment(experiment_name, random=args.random)
    print("Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc, test_acc))
    # train_acc, test_acc = run_experiment(experiment_name, random=True)
    # print("Random Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc, test_acc))
    print("\n")