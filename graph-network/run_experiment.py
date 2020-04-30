import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from Experiments import Experiments, Experiment
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('experiment', default=None,
                    help='Select experiment [apicall|link|typeuse|varuse|fname|nodetype]')
args = parser.parse_args()

"""
apicall creates an experiment where we try to predict the exstense of "next call" link between nodes. 
    Currently, the process is trained using negative sampling procedure and additional classifier. Negative sampling 
    is done uniformly form the collection of all the rest of the nodes. The number of negative samples is the batch is 
    equal to the number of positive samples. The embeddigns themselves are not trained, only weights of classifier are 
    updated. 
"""

# TODO
# can maybe the scores I'm getting are overfit?
# Need to generate not random edges, but negative edges for the same destination. There can be a situation where
# I'm getting good results simply because negative edges do not touch positive nodes

# BASE_PATH = "/home/ltv/data/local_run/graph-network/GAT-2020-03-24-03-37-36-421131" # trained on node types
# BASE_PATH = "/home/ltv/dev/method-embedding/graph-network/models/GAT-2020-04-24-00-21-11-377353" # trained on function names
# BASE_PATH = "/home/ltv/dev/method-embedding/graph-network/models/GAT-2020-04-23-22-33-41-890539" # trained on function names with classifier
BASE_PATH = "/home/ltv/dev/method-embedding/graph-network/models/GAT-2020-04-24-19-07-55-139367" # trained on next call
# BASE_PATH = "/home/ltv/dev/method-embedding/graph-network/models/GAT-2020-04-24-22-12-19-718445" # trained on variables
# API_SEARCH = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/04_api_sequence_calls/flat_calls.csv"
API_SEQ = "/home/ltv/data/datasets/source_code/python-source-graph/04_api_sequence_calls/flat_calls.csv"
VAR_USE = "/home/ltv/data/datasets/source_code/python-source-graph/03_variables_on_functions/node_to_var.csv"

e = Experiments(base_path=BASE_PATH,
                api_seq_path=API_SEQ,
                type_use_path=None, #not needed
                node_type_path=None, #not needed
                variable_use_path=VAR_USE, #not needed
                function_name_path=None
                )

# EXPERIMENT_NAME = "fname"
EXPERIMENT_NAME = args.experiment

experiment = e[EXPERIMENT_NAME]

#%%
####################################################################
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report, accuracy_score
#
# lr = LogisticRegression(max_iter=1000)
#
# experiment.get_training_data()
# X_train, y_train = experiment._embed(experiment.X_train), experiment.y_train
# X_test, y_test = experiment._embed(experiment.X_test), experiment.y_test
#
# lr.fit(X_train, y_train)
#
# print(pandas.DataFrame(classification_report(y_test, lr.predict(X_test), output_dict=True)))
#
# print(accuracy_score(y_test, lr.predict(X_test)))
#
# # print(test_positive_dst.size / (test_positive_dst.size + test_negative_dst.size))

#####################################################################

from classifiers import LRClassifier, NNClassifier, ElementPredictor, NodeClassifier
import tensorflow as tf

if EXPERIMENT_NAME in {'link', 'apicall', 'typeuse'}:
    # TODO
    # these three do not work ocrrectly, check old version
    clf = NNClassifier(experiment.embed_size)
elif EXPERIMENT_NAME in {'varuse', 'fname'}:
    clf = ElementPredictor(experiment.embed_size, experiment.unique_elements, 100)
elif EXPERIMENT_NAME in {'nodetype'}:
    clf = NodeClassifier(experiment.embed_size, experiment.unique_elements)

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
        # print(": )", labels.shape, predictions.shape)
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

print(f"\n\n\nExperiment name: {EXPERIMENT_NAME}")

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


    for batch_ind, batch in enumerate(experiment.train_batches()):
        train_step(batch)

        if batch_ind % 500 == 0:

            for batch in experiment.test_batches():
                test_step(batch)

            template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
            print(template.format(epoch+1,
                                  train_loss.result(),
                                  train_accuracy.result()*100,
                                  test_loss.result(),
                                  test_accuracy.result()*100))
