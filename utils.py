import os, warnings, re
import numpy as np
import torch.utils.data as data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        transformed = []
        for string in X:
            transformed.append(clean_str(string))
        return transformed


class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        from nltk import word_tokenize
        tokenized = []
        for text in X:
            tokenized.append(word_tokenize(text))

        return tokenized


class PadSequencer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        lengths = [len(text) for text in X]
        self.max_len = max(lengths)

        new_X = []

        for text in X:
            new = text
            while len(text) < self.max_len:
                new.append('<PAD>')
            new_X.append(new)

        return new_X


class Indexizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.vocab = []
        for text in X:
            for w in text:
                if w not in self.vocab:
                    self.vocab.append(w)
        self.idx2w = dict(enumerate(self.vocab))
        self.w2idx = {w: i for (i, w) in self.idx2w.items()}

        new = []
        for text in X:
            new.append([self.w2idx[w] for w in text])

        return np.array(new)


class DataLoader:
    '''
    Basic data loader to load in a directory or file of data as well as useful helper functions
    '''

    def __init__(self, directory, batch_size=32, shuffle=True, num_workers=2):
        self.dir, self.batch_size = directory, batch_size

        # Exception handling for various input formats
        if type(directory) == list:
            self._raw_data, self.labels = self._process_list(directory)
        elif os.path.isdir(directory):
            self._raw_data, self.labels = self._process_dir(directory)
        elif os.path.isfile(directory):
            self._raw_data, self.labels = self._process_file(directory)
        assert self._raw_data is not None, \
            "Error: Something's gone wrong, please check the contents of {}\n" \
            "The format must be either a directory of files, a directory of directories," \
            "or a file.".format(directory)

        # Index tensorization
        text_preprocessor = Pipeline([
            ('clean', Cleaner()),
            ('tokenize', Tokenizer()),
            ('pad', PadSequencer()),
            ('idx', Indexizer())
        ])
        lb = LabelEncoder()
        one_hot = OneHotEncoder()

        self.X = text_preprocessor.fit_transform(self._raw_data)
        self.y = lb.fit_transform(self.labels)
        # self.y = one_hot.fit_transform(lb.fit_transform(self.labels).reshape(-1, 1)).todense()

        self.V = text_preprocessor.named_steps['idx'].vocab
        self.idx2w = text_preprocessor.named_steps['idx'].idx2w
        self.w2idx = text_preprocessor.named_steps['idx'].w2idx

    def _process_dir(self, directory):
        texts = []

        for file in os.listdir(directory):
            if os.path.isdir(file):
                subdir_texts = self._process_dir(file)
                for text in subdir_texts:
                    texts.append(text)
            elif os.path.isfile(file):
                texts.append(open(file, 'rb').read().decode('utf-8', 'ignore'))
            else:
                continue

        return texts

    def _process_file(self, file):
        text = []
        try:
            text.append(open(file, 'rb').read().decode('utf-8', 'ignore'))
        except FileNotFoundError:
            print("Error: {} not found".format(file))

        return text

    def _process_list(self, l):
        '''
        Process list of lists, where each l[0] = str and l[1] = label
        Only to be used for testing
        :param l:
        '''
        X = [x[0] for x in l]
        y = [y[1] for y in l]
        return X, y

    def batch_data(self, X, y, minibatch_size=32):
        n = len(X)
        idxs = np.random.permutation(n)
        minibatches = []

        for i in range(minibatch_size, n, minibatch_size):
            minibatches.append([X[idxs[i - minibatch_size:i]], y[idxs[i - minibatch_size:i]]])

    def __len__(self):
        warnings.warn("usage of len on DataLoader returns length of vocabulary,"
                      "for number of batches please use len(DataLoader.data),"
                      "for length of the raw dataset use len(DataLoader._raw_data)", Warning)
        return len(self.V)

    def __str__(self):
        return "Dataset built off of: {}\nBatch size:  {}\nNumber of batches: {}".format(
            self.dir, self.batch_size, len(self.data)
        )

    def __getitem__(self, index):
        return self.X_[index], self.y_[index]