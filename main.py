import argparse
import sys
import numpy as np
from utils import DataLoader
from model import TextCNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-D'
        '--datafile',
        help='File or directory containing data',
        required=True,
        dest='datafile')
    parser.add_argument(
        '-b',
        '--batch_size',
        help='Minibatch size',
        default=32,
        dest='batch_size')
    parser.add_argument(
        '-E',
        '--num_epochs',
        help='Number of training epochs',
        default=5,
        dest='num_epochs')
    parser.add_argument(
        '-lr',
        '--learning_rate',
        help='Learning rate for optimizer',
        default=0.01,
        dest='lr')
    parser.add_argument(
        '-d',
        '--dropout',
        help='Dropout rate for regularization',
        default=0.5,
        dest='dropout')
    parser.add_argument(
        '-e',
        '--embed_dim',
        help='Dimension of embeddings (if larger than 300 cannot use pretrained',
        default=300,
        dest='embed_dim')
    parser.add_argument(
        '-K',
        '--kernel_num',
        help='Number of kernels per size',
        default=100,
        dest='kernel_num')
    parser.add_argument(
        '-k',
        '--kernel_sizes',
        help='Various filter sizes',
        default='3,4,5',
        dest='kernel_sizes')
    parser.add_argument(
        '-m',
        '--multichannel',
        help='Whether to use multichannel model',
        default=False,
        dest='multichannel')
    parser.add_argument(
        '-s',
        '--static',
        help='Whether to use static word embeddings',
        default=False,
        dest='static')
    parser.add_argument(
        '-p',
        '--pretrained',
        help='Path to pretrained word vectors,'
        'set if you want to use pretrained',
        default=None,
        dest='pretrained')
    parser.add_argument(
        '-t',
        '--test',
        help='Inference on pretrained model or not',
        default=False,
        dest='test')
    parser.add_argument(
        '-v',
        '--val',
        help='Validation mode',
        default=False,
        dest='val')
    parser.add_argument(
        '-n',
        '--experiment_num',
        help='Number of experiment',
        default=1,
        dest='experiment_num')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    dl = DataLoader(args['datafile'])
    args['multichannel'] = bool(args['multichannel'])
    args['static'] = bool(args['static'])
    args['test'] = bool(args['test'])
    args['val'] = bool(args['val'])
    args['num_epochs'] = int(args['num_epochs'])
    args['batch_size'] = int(args['batch_size'])
    args['vocab_size'] = len(dl)
    args['class_num'] = len(set(dl.y))
    arg_names = ['num_epochs', 'lr', 'dropout', 'embed_dim',
                 'kernel_num', 'kernel_sizes', 'static',
                 'multichannel', 'batch_size', 'vocab_size',
                 'class_num']
    clf_args = {x: y for (x, y) in args.items() if x in arg_names}

    X_train, X_test, y_train, y_test = train_test_split(
        dl.X, dl.y, random_state=42, test_size=0.33)

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, random_state=42, test_size=0.2)

    clf = TextCNNClassifier(**clf_args)

    if args['test']:
        clf.load(
            'model_saves/experiment_{}'.format(int(args['experiment_num'])))

        preds = []  # Unwrap model inference for better memory allocation

        for i in range(X_test.shape[0]):
            pred = clf.model_.forward(np.array([X_test[i, :]])).data.numpy()
            preds.append(pred)

        preds = np.array(preds).squeeze(axis=1)

        rounded_preds = np.argmax(preds, axis=1)

        print("Experiment {} Accuracy: {}".format(
            int(args['experiment_num']), accuracy_score(y_test, rounded_preds)))

        print("Finding Test Examples that Maximize Classes...")

        max_indices = np.argmax(preds, axis=0)

        for i in range(len(max_indices)):
            sent_idxs = X_test[max_indices[i], :]
            sent_str = ' '.join([dl.idx2w[j]
                                 for j in sent_idxs if j != dl.w2idx['<PAD>']])
            print('class {} (true {}): {}'.format(dl._lb.inverse_transform(
                i), dl._lb.inverse_transform(y_test[max_indices[i]]), sent_str))
        sys.exit()
    elif args['val']:
        clf.load(
            'model_saves/experiment_{}'.format(int(args['experiment_num'])))

        preds = []

        for i in range(X_dev.shape[0]):
            pred = clf.model_.forward(np.array([X_dev[i, :]])).data.numpy()
            preds.append(pred)

        preds = np.array(preds).squeeze(axis=1)

        rounded_preds = np.argmax(preds, axis=1)
        print(
            "Validation accuracy: {}".format(
                accuracy_score(
                    y_dev,
                    rounded_preds)))

        print("Finding Val Examples that Maximize Classes...")

        max_indices = np.argmax(preds, axis=0)

        for i in range(len(max_indices)):
            sent_idxs = X_dev[max_indices[i], :]
            sent_str = ' '.join([dl.idx2w[j]
                                 for j in sent_idxs if j != dl.w2idx['<PAD>']])
            print('class {} (true {}): {}'.format(dl._lb.inverse_transform(
                i), dl._lb.inverse_transform(y_dev[max_indices[i]]), sent_str))
        sys.exit()

    if args['pretrained'] is not None:
        clf.load_pretrained(args['pretrained'], dl)

    clf.fit(X_train, y_train, dl)

    preds = []

    for i in range(X_dev.shape[0]):
        pred = clf.model_.forward(np.array([X_dev[i, :]])).data.numpy()
        preds.append(pred)

    preds = np.array(preds).squeeze(axis=1)

    rounded_preds = np.argmax(preds, axis=1)

    print(
        "Validation accuracy: {}".format(
            accuracy_score(
                y_dev,
                rounded_preds)))

    clf.save('model_saves/experiment_{}'.format(int(args['experiment_num'])))

    print(
        "Model saved to: model_saves/experiment_{}".format(int(args['experiment_num'])))

    print("Finding Val Examples that Maximize Classes...")

    max_indices = np.argmax(preds, axis=0)

    for i in range(len(max_indices)):
        sent_idxs = X_dev[max_indices[i], :]
        sent_str = ' '.join([dl.idx2w[j]
                             for j in sent_idxs if j != dl.w2idx['<PAD>']])
        print('class {} (true {}): {}'.format(dl._lb.inverse_transform(
            i), dl._lb.inverse_transform(y_dev[max_indices[i]]), sent_str))
