import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Experiments import Experiments, Experiment
import argparse
from classifiers import LRClassifier, NNClassifier, ElementPredictor, NodeClassifier
import tensorflow as tf

# tf.get_logger().setLevel('ERROR')
import numpy as np
from copy import deepcopy
from ast import literal_eval

link_prediction_experiments = ['link', 'apicall', 'typeuse', 'typelink', 'typelink_tt']
name_prediction_experiments = ['varuse', 'fname']
node_classification_experiments = ['nodetype', "typeann"]
all_experiments = link_prediction_experiments + name_prediction_experiments + node_classification_experiments


def run_experiment(e, experiment_name, args):
    experiment = e[experiment_name]

    ma_train = 0.
    ma_test = 0.
    ma_alpha = 2 / (10 + 1)

    if args.random:
        experiment.embed.e = deepcopy(experiment.embed.e)  # ????
        experiment.embed.e = np.random.randn(experiment.embed.e.shape[0], experiment.embed.e.shape[1])

    if args.test_embedder:
        # this does not really show anything new compared with random initialization. need to compare
        # against Node2Vec or KG embeddings
        experiment.embed.e = tf.Variable(
            initial_value=np.random.randn(experiment.embed.e.shape[0], experiment.embed.e.shape[1]),
            trainable=True
        )

    if experiment_name in link_prediction_experiments:
        clf = NNClassifier(experiment.embed_size,
                           h_size=literal_eval(args.link_predictor_h_size))
    elif experiment_name in name_prediction_experiments:
        clf = ElementPredictor(experiment.embed_size, experiment.unique_elements, args.name_emb_dim,
                               h_size=args.element_predictor_h_size)
    elif experiment_name in node_classification_experiments:
        clf = NodeClassifier(experiment.embed_size, experiment.unique_elements,
                             h_size=literal_eval(args.node_classifier_h_size))
    else:
        raise ValueError(
            f"Unknown experiment: {type}. The following experiments are available: [{'|'.join(all_experiments)}].")

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

    args.epochs = 500

    tests = []

    for epoch in range(args.epochs):
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="""
        Currently, the process is trained using negative sampling procedure and additional classifier. Negative sampling 
        is done uniformly form the collection of all the rest of the nodes. The number of negative samples is the batch is 
        equal to the number of positive samples. The embeddings themselves are not trained, only weights of classifier are 
        updated. 
    """)
    parser.add_argument('--experiment', default=all_experiments,
                        help=f'Select experiment [{"|".join(all_experiments)}]')
    parser.add_argument("--base_path", default=None, help="path to the trained GNN model")
    parser.add_argument("--api_seq", default=None, help="")
    parser.add_argument("--var_use", default=None, help="")
    parser.add_argument("--type_ann", default=None, help="")
    parser.add_argument("--type_link", default=None, help="")
    parser.add_argument("--type_link_train", default=None, help="")
    parser.add_argument("--type_link_test", default=None, help="")
    parser.add_argument("--epochs", default=500, type=int, help="")
    parser.add_argument("--name_emb_dim", default=100, type=int, help="")
    parser.add_argument("--element_predictor_h_size", default=50, type=int, help="")
    parser.add_argument("--link_predictor_h_size", default="[20]", type=str, help="")
    parser.add_argument("--node_classifier_h_size", default="[30,15]", type=str, help="")
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--test_embedder', action='store_true')
    args = parser.parse_args()

    e = Experiments(base_path=args.base_path,
                    api_seq_path=args.api_seq,
                    type_link_path=args.type_link,
                    type_link_train_path=args.type_link_train,
                    type_link_test_path=args.type_link_test,
                    variable_use_path=args.var_use,  # not needed
                    function_name_path=None,
                    type_ann=args.type_ann,
                    gnn_layer=-1,
                    )

    experiments = args.experiment.split(",")

    for experiment_name in experiments:
        print(f"\n{experiment_name}:")
        try:
            train_acc, test_acc = run_experiment(e, experiment_name, args)
            print("Train Accuracy: {:.4f}, Test Accuracy: {:.4f}".format(train_acc, test_acc))
        except ValueError as err:
            print(err)
        print("\n")
