import os

from sklearn import metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from Experiments import Experiments, Experiment
import argparse
from classifiers import LRClassifier, NNClassifier, ElementPredictor, NodeClassifier, ElementPredictorWithSubwords
import tensorflow as tf
import tensorflow_addons as tfa

# tf.get_logger().setLevel('ERROR')
import numpy as np
from copy import deepcopy
from ast import literal_eval

link_prediction_experiments = ['link', 'apicall', 'typeuse', 'typelink', 'typelink_tt']
name_prediction_experiments = ['varuse', 'fname']
name_subword_predictor = ['typeann_name']
node_classification_experiments = ['nodetype', "typeann"]
all_experiments = link_prediction_experiments + name_prediction_experiments + node_classification_experiments + name_subword_predictor


class Tracker:
    def __init__(self, inverted_index=None):
        self.all_true = []
        self.all_estimated = []
        self.all_emb = []
        self.inv_index = inverted_index

    def add(self, embs, pred, true):
        self.all_true.append(true)
        self.all_estimated.append(pred)
        self.all_emb.append(embs)

    def decode_label_names(self, ids):
        assert self.inv_index is not None, "Need inverted index"
        return list(map(lambda x: self.inv_index[x], ids))

    @property
    def true_labels(self):
        return np.concatenate(self.all_true, axis=0).reshape(-1,).tolist()

    @property
    def true_label_names(self):
        return self.decode_label_names(self.true_labels)

    @property
    def pred_labels(self):
        return np.argmax(np.concatenate(self.all_estimated, axis=0), axis=-1).reshape(-1, ).tolist()

    @property
    def pred_label_names(self):
        return self.decode_label_names(self.pred_labels)

    @property
    def pred_scores(self):
        return np.concatenate(self.all_estimated, axis=0)

    @property
    def embeddings(self):
        return np.concatenate(self.all_emb, axis=0)

    def clear(self):
        self.all_true.clear()
        self.all_estimated.clear()
        self.all_emb.clear()

    def save_embs_for_tb(self, save_name):
        assert self.inv_index is not None, "Cannot export for tensorboard without metadata"
        np.savetxt(f"{save_name}_embeddings.tsv", self.embeddings, delimiter="\t")
        with open(f"{save_name}_meta.tsv", "w") as meta_sink:
            for label in list(map(lambda x: self.inv_index[x], self.true_labels)):
                meta_sink.write(f"{label}\n")

    def save_umap(self, save_name):
        type_freq = {
            "str": 532,
            "Optional": 232,
            "int": 206,
            "Any": 171,
            "Union": 156,
            "bool": 143,
            "Callable": 80,
            "Dict": 77,
            "bytes": 58,
            "float": 48
        }
        from umap import UMAP
        import matplotlib.pyplot as plt
        # plt.rcParams.update({'font.size': 5})
        reducer = UMAP(50)
        embedding = reducer.fit_transform(self.embeddings)

        labels = list(map(lambda x: self.inv_index[x], self.true_labels))
        unique_labels = sorted(list(set(labels)))

        plt.figure(figsize=(6,6))
        legend = []
        for label in unique_labels:
            if label not in type_freq:
                continue
            xs = []
            ys = []
            for lbl, (x, y) in zip(labels, embedding):
                if lbl == label:
                    xs.append(x)
                    ys.append(y)
            plt.scatter(xs, ys, 1.)
            legend.append(label)
        plt.axis('off')
        plt.legend(legend)
        plt.savefig(f"{save_name}_umap.pdf")
        plt.close()
        # plt.show()



    def get_metrics(self):
        all_true = self.true_labels
        all_scores = self.pred_scores

        metric_dict = {}

        for k in [1,3,5]:
            metric_dict[f"Acc@{k}"] = metrics.top_k_accuracy_score(y_true=all_true, y_score=all_scores, k=k, labels=list(range(all_scores.shape[1])))

        return metric_dict

    def save_confusion_matrix(self, save_path):
        estimate_confusion(
            self.pred_label_names,
            self.true_label_names,
            save_path=save_path
        )


def estimate_confusion(pred, true, save_path):
    pred_filtered = pred
    true_filtered = true

    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    # plt.rcParams.update({'font.size': 50})

    labels = sorted(list(set(true_filtered + pred_filtered)))
    label2ind = dict(zip(labels, range(len(labels))))

    confusion = np.zeros((len(labels), len(labels)))

    for pred, true in zip(pred_filtered, true_filtered):
        confusion[label2ind[true], label2ind[pred]] += 1

    norm = np.array([x if x != 0 else 1. for x in np.sum(confusion, axis=1)]).reshape(-1,1)
    confusion /= norm


    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(confusion, interpolation='nearest', cmap=cm.Blues)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{confusion[i, j]: .2f}",
                           ha="center", va="center", color="w", fontsize="xx-small")

    # ax.set_title("Confusion matrix for Python type prediction")
    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_path, "confusion.pdf"))
    plt.close()


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
    elif experiment_name in name_subword_predictor:
        clf = ElementPredictorWithSubwords(experiment.embed_size, experiment.num_buckets, args.name_emb_dim,
                               h_size=args.element_predictor_h_size)
    else:
        raise ValueError(
            f"Unknown experiment: {type}. The following experiments are available: [{'|'.join(all_experiments)}].")

    # clf.compile(optimizer='adam',
    #             loss='sparse_categorical_crossentropy')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer = tfa.optimizers.lazy_adam.LazyAdam(learning_rate=0.001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # @tf.function
    def train_step(batch, tracker=None):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = clf(**batch, training=True)
            loss = loss_object(batch["y"], predictions)
            gradients = tape.gradient(loss, clf.trainable_variables)
            optimizer.apply_gradients(zip(gradients, clf.trainable_variables))

        if tracker is not None:
            tracker.add(batch["x"], predictions.numpy(), batch["y"])

        train_loss(loss)
        train_accuracy(batch["y"], predictions)

    # @tf.function
    def test_step(batch, tracker=None):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = clf(**batch, training=False)
        t_loss = loss_object(batch["y"], predictions)

        if tracker is not None:
            tracker.add(batch["x"], predictions.numpy(), batch["y"])

        test_loss(t_loss)
        test_accuracy(batch["y"], predictions)

    trains = []
    tests = []
    metrics = []

    test_tracker = Tracker(inverted_index=experiment.inv_index if hasattr(experiment, "inv_index") else None)

    for epoch in range(args.epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for batch_ind, batch in enumerate(experiment.train_batches()):
            train_step(batch)

        if epoch % 1 == 0:

            test_tracker.clear()

            for batch in experiment.test_batches():
                test_step(batch, tracker=test_tracker)

            ma_train = train_accuracy.result() * 100 * ma_alpha + ma_train * (1 - ma_alpha)
            ma_test = test_accuracy.result() * 100 * ma_alpha + ma_test * (1 - ma_alpha)
            trains.append(ma_train)
            tests.append(ma_test)
            metrics.append(test_tracker.get_metrics())

            # template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}, Average Test {:.4f}'
            # print(template.format(epoch+1,
            #                       train_loss.result(),
            #                       train_accuracy.result()*100,
            #                       test_loss.result(),
            #                       test_accuracy.result()*100,
            #                       ma_test))

    # plot confusion matrix
    if hasattr(experiment, "inv_index") and args.out_path is not None:
        test_tracker.save_confusion_matrix(save_path=args.out_path)

    if hasattr(experiment, "inv_index") and args.emb_out:
        test_tracker.save_embs_for_tb(save_name=os.path.join(args.out_path, "tb"))
        test_tracker.save_umap(save_name=os.path.join(args.out_path, "tb"))

    print(metrics[tests.index(max(tests))])

    # ma_train = train_accuracy.result() * 100 * ma_alpha + ma_train * (1 - ma_alpha)
    # ma_test = test_accuracy.result() * 100 * ma_alpha + ma_test * (1 - ma_alpha)

    import matplotlib.pyplot as plt
    plt.plot(trains)
    plt.plot(tests)
    plt.legend(["Train", "Test"])
    # plt.savefig(os.path.join(args.base_path, f"{args.experiment}.png"))
    plt.show()

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
    parser.add_argument("--out_path", default=None, type=str, help="")
    # parser.add_argument("--confusion_out_path", default=None, type=str, help="")
    parser.add_argument("--trials", default=1, type=int, help="")
    parser.add_argument("--emb_out", default=False, action="store_true", help="")
    parser.add_argument("--only_popular_types", default=False, action="store_true", help="")
    parser.add_argument("--popular_types", default="str,Optional,int,Any,Union,bool,Other,Callable,Dict,bytes,float,Description,List,Sequence,Namespace,T,Type,object,HTTPServerRequest,Future,Matcher", help="")
    # parser.add_argument("--emb_out", default=None, type=str, help="")
    parser.add_argument("--embeddings", default=None)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--test_embedder', action='store_true')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.dirname(args.embeddings)

    if args.popular_types is not None:
        args.popular_types = args.popular_types.split(",")

    print(args.__dict__)

    e = Experiments(
        base_path=args.base_path, api_seq_path=args.api_seq, type_link_path=args.type_link,
        type_link_train_path=args.type_link_train, type_link_test_path=args.type_link_test,
        variable_use_path=args.var_use,  # not needed
        function_name_path=None,
        type_ann=args.type_ann, gnn_layer=-1, embeddings_path=args.embeddings,
        only_popular_types=args.only_popular_types, popular_types=args.popular_types
    )

    experiments = args.experiment.split(",")

    for experiment_name in experiments:
        for trial in range(args.trials):
            print(f"\n{experiment_name}, trial {trial}:")
            try:
                train_acc, test_acc = run_experiment(e, experiment_name, args)
                print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
            except ValueError as err:
                print(err)
            print("\n")
