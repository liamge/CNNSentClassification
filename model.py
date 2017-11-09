import numpy as np
import torch
import warnings
import gensim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted
from utils import DataLoader


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()

        # Command line arguments to be parsed into a dictionary before passing to object
        self.args = args

        assert not (self.args['static'] and self.args['multichannel']), "Error: Cannot be both static and multichannel"

        # Random seed
        torch.manual_seed(42)

        # Convolutional Parameters
        # Input channels
        if args['multichannel']:
            Cin = 2
        else:
            Cin = 1

        # Output channels = number of kernels for each
        Cout = args['kernel_num']

        # Kernels
        Ks = [int(k) for k in self.args['kernel_sizes'].split(',')]

        # Embedding dimensionality
        d = args['embed_dim']

        # Vocabulary
        V = args['vocab_size']

        # Number of classes
        num_classes = args['class_num']

        # Embedding layer
        self.embed = nn.Embedding(V, d)

        # Static Embedding layer
        if self.args['static'] or self.args['multichannel']:
            self.static_embed = nn.Embedding(V, d)
            self.static_embed.weight.requires_grad = False

        # Convolutional layer
        self.convs = nn.ModuleList([nn.Conv2d(Cin, Cout, (k, d)) for k in Ks])

        # Dropout layer
        self.dropout = nn.Dropout(args['dropout'])

        # Output layer
        self.linear = nn.Linear(len(Ks) * args['kernel_num'], num_classes)

    def forward(self, x):
        if self.args['multichannel']:
            x_static = self.static_embed(x)
            x_static = torch.unsqueeze(x_static, 1)
            x_dynamic = self.embed(x)
            x_dynamic = torch.unsqueeze(x_dynamic, 1)
            x = torch.cat([x_dynamic, x_static], 1)  # (minibatch x 2 x sentence_len x embed_dim)
        elif self.args['static']:
            x = self.static_embed(x)
            x = torch.unsqueeze(x, 1)  # (minibatch x 1 x sentence_len x embed_dim
        else:
            x = self.embed(x)  # (minibatch x sentence_len x embed_dim)
            x = torch.unsqueeze(x, 1)  # (minibatch x input_channels x sentence_len x embed_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [N x Cout x output of convolving kernel k] * len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [N x Cout] * len(Ks)
        x = torch.cat(x, 1)  # (N, Cout * len(Ks))
        x = self.dropout(x)
        x = nn.LogSoftmax()(self.linear(x))
        return x

    def load_word_vectors(self, path, data):
        '''
        Loads pretrained word vectors and rewrites embedding layer weights with them
        :param path: Path to binary word vectors
        :param data: DataLoader object for a corpus of data
        :return: None
        '''
        self.word_vecs = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True, limit=500)

        self.embed.padding_idx = data.w2idx['<PAD>']

        # Load vocab
        V = data.V
        W = np.zeros((self.args['vocab_size'], self.args['embed_dim']))
        for (i, w) in enumerate(V):
            if w in self.word_vecs.vocab:
                W[i, :] = self.word_vecs[w]
            else:
                W[i, :] = np.random.uniform(-0.25, 0.25, self.args['embed_dim'])

        self.embed.weight.data.copy_(torch.from_numpy(W).float())
        if self.args['static'] or self.args['multichannel']:
            self.static_embed.weight.data.copy_(torch.from_numpy(W).float())


class TextCNNClassifier(BaseEstimator, ClassifierMixin):
    # Wrapper for interaction with sklearn API
    def __init__(self, data, num_epochs=5, lr=0.001, dropout=0.5,
                 embed_dim=300, kernel_num=100, kernel_sizes='3,4,5',
                 static=False, multichannel=False, pretrained=None,
                 vocab_size=50000, class_num=2, word_freq_threshold=1):

        # For get_params()/set_params()
        self.num_epochs = num_epochs
        self.lr = lr
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.static = static
        self.multichannel = multichannel
        self.pretrained = pretrained
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.word_freq_threshold = word_freq_threshold
        self.args_ = self.get_params()

        # Dataloader
        self.data_ = data
        self.args_['vocab_size'] = len(self.data_)

        # Initialize model
        self.model_ = TextCNN(args=self.args_)

        # Load pretrained vectors
        if self.pretrained is not None:
            self.model_.load_word_vectors(self.args_['pretrained'], self.data_)

    def fit(self, X, y, plot=False):
        '''
        Run training on X and y
        :param X: List of indices representing words
        :param y: list of labels
        :return: None
        '''
        self.X_ = Variable(torch.from_numpy(X))
        self.y_ = Variable(torch.from_numpy(y))

        # batches = self.data_.batch_data(X, y)

        # Train the model
        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam([p for p in self.model_.parameters() if p.requires_grad], lr=self.args_['lr'])

        losses = []

        '''
        # Doesn't need dev iter b/c wrapper can use cross_val_score
        for i in range(self.num_epochs):
            print('Epoch {}...'.format(i))
            for batch in batches:
                # Zero out cumulative gradients
                optimizer.zero_grad()

                x, y = batch

                logits = self.model_.forward(X)
                loss = criterion(logits, y)

                losses.append(loss.data.numpy()[0])

                # Print continuous loss while training
                print('Batch Loss: {}'.format(loss.data.numpy()[0]))

                # Backprop
                loss.backward()

                # Weight update
                optimizer.step()
        '''

        # Testing loop
        for i in range(self.num_epochs):
            print('Epoch {}...'.format(i))
            optimizer.zero_grad()
            logits = self.model_.forward(self.X_)
            loss = criterion(logits, self.y_)
            losses.append(loss.data.numpy()[0])
            print('Batch Loss: {}'.format(loss.data.numpy()[0]))
            loss.backward()
            optimizer.step()

        print('Successfully trained!')

        if plot:
            plt.plot(losses)

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        return np.argmax(self.model_.forward(Variable(torch.from_numpy(dl.X))).data.numpy(), axis=0)

    def save(self, path):
        '''
        Serializes model
        :param path: Path to serialized model
        :return: None
        '''
        torch.save(self.model_.state_dict(), path)

    def load(self, path):
        self.model_.load_state_dict(torch.load(path))

    def __call__(self, input):
        return self.predict(input)

x = Variable(torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]))
y = Variable(torch.LongTensor([0, 1]))
args = {'num_epochs': 5, 'lr': 0.001, 'dropout': 0.5,
        'embed_dim': 300, 'kernel_num': 100, 'kernel_sizes': '2,3,4',
        'static': False, 'multichannel': False, 'pretrained': None,
        'vocab_size': 50000, 'class_num': 2}
