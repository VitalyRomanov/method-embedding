from Experiments import Experiments, Experiment

BASE_PATH = "/home/ltv/data/local_run/graph-network/GAT-2020-03-24-03-37-36-421131"
# API_SEARCH = "/Volumes/External/datasets/Code/source-graphs/python-source-graph/04_api_sequence_calls/flat_calls.csv"
API_SEQ = "/home/ltv/data/datasets/source_code/python-source-graph/04_api_sequence_calls/flat_calls.csv"

e = Experiments(base_path=BASE_PATH,
                api_seq_path=API_SEQ,
                type_use_path=None,
                node_type_path=None,
                variable_use_path=None,
                function_name_path=None
                )

EXPERIMENT_NAME = "fname"

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
    clf = LRClassifier(experiment.embed_size)
elif EXPERIMENT_NAME in {'varuse', 'fname'}:
    clf = ElementPredictor(experiment.embed_size, experiment.unique_elements, 100)
elif EXPERIMENT_NAME in {'nodetype'}:
    clf = NodeClassifier(experiment.embed_size, experiment.unique_elements)

# clf.compile(optimizer='adam',
#             loss='sparse_categorical_crossentropy')

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# @tf.function
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

# @tf.function
def test_step(batch):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = clf(**batch, training=False)
    t_loss = loss_object(batch["y"], predictions)

    test_loss(t_loss)
    test_accuracy(batch["y"], predictions)

EPOCHS = 500

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


    for batch in experiment.train_batches():
        train_step(batch)

        for batch in experiment.test_batches():
            test_step(batch)

        template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()*100,
                              test_loss.result(),
                              test_accuracy.result()*100))
