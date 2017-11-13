import numpy as np
import torch
import gensim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


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

        # Xavier Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))

        # Dropout layer
        self.dropout = nn.Dropout(args['dropout'])

        # Output layer
        self.linear = nn.Linear(len(Ks) * args['kernel_num'], num_classes)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = Variable(torch.from_numpy(x))
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
        word_vecs = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)

        self.embed.padding_idx = data.w2idx['<PAD>']

        # Load vocab
        V = data.V
        W = np.zeros((self.args['vocab_size'], self.args['embed_dim']))
        for (i, w) in enumerate(V):
            if w in word_vecs.vocab:
                W[i, :] = word_vecs[w]
            else:
                W[i, :] = np.random.uniform(-0.25, 0.25, self.args['embed_dim'])

        self.embed.weight.data.copy_(torch.from_numpy(W).float())
        if self.args['static'] or self.args['multichannel']:
            self.static_embed.weight.data.copy_(torch.from_numpy(W).float())


class TextCNNClassifier(BaseEstimator, ClassifierMixin):
    # Wrapper for interaction with sklearn API
    def __init__(self, num_epochs=5, lr=0.001, dropout=0.5,
                 embed_dim=300, kernel_num=100, kernel_sizes='3,4,5',
                 static=False, multichannel=False, pretrained=None,
                 vocab_size=100000, class_num=2, batch_size=32):

        # For get_params()/set_params()
        self.num_epochs = num_epochs
        self.lr = lr
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.static = static
        self.multichannel = multichannel
        self.vocab_size = vocab_size
        self.class_num = class_num
        self.batch_size = batch_size
        self.args_ = self.get_params()

        # Initialize model
        self.model_ = TextCNN(args=self.args_)

    def fit(self, X, y, data, plot=False):
        '''
        Run training on X and y
        :param X: List of indices representing words
        :param y: list of labels
        :return: None
        '''
        # Train the model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam([p for p in self.model_.parameters() if p.requires_grad], lr=self.args_['lr'])

        self.losses = []

        # Doesn't need dev iter b/c wrapper can use cross_val_score
        for i in range(self.num_epochs):
            batches = list(data.batch_data(X, y, minibatch_size=self.batch_size))[:-1]
            print('Epoch {}...'.format(i))
            print("#####################")
            batch_loss = []
            for batch in batches:
                # Zero out cumulative gradients
                optimizer.zero_grad()

                x_, y_ = batch
                batch_x, batch_y = Variable(torch.from_numpy(x_)), Variable(torch.from_numpy(y_), requires_grad=False)

                logits = self.model_.forward(batch_x)
                loss = criterion(logits, batch_y)

                self.losses.append(loss.data.numpy()[0])
                batch_loss.append(loss.data.numpy()[0])

                # Print continuous loss while training
                print("Batch Loss: {}".format(loss.data.numpy()[0]), end='\r')

                # Backprop
                loss.backward()

                # Weight update
                optimizer.step()

            print('\n')
            print("Epoch {} Loss: {}".format(i, np.mean(batch_loss)))
            print("#####################")

        print('Successfully trained!')

        if plot:
            plt.plot(self.losses)

        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['losses'])

        return np.argmax(self.model_.forward(Variable(torch.from_numpy(X))).data.numpy(), axis=1)

    def save(self, path):
        '''
        Serializes model
        :param path: Path to serialized model
        :return: None
        '''
        torch.save(self.model_.state_dict(), path)

    def load(self, path):
        self.losses = [] # Define to allow inference

        self.model_.load_state_dict(torch.load(path))

    def load_pretrained(self, path, dataloader):
        '''
        Loads pretrained word vectors
        :param path: Path to pretrained word vectors
        :param dataloader: DataLoader object
        '''
        self.model_.load_word_vectors(path, dataloader)

    def __call__(self, input):
        return self.predict(input)
