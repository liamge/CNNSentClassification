import os
import warnings
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


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

    def __init__(self, directory):
        self.dir = directory

        # Exception handling for various input formats
        if isinstance(directory, list):
            raw_data, labels = self._process_list(directory)
        elif os.path.isdir(directory):
            raw_data, labels = self._process_dir(directory)
        elif os.path.isfile(directory):
            raw_data, labels = self._process_file(directory)
        assert raw_data is not None, \
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
        self._lb = LabelEncoder()

        self.X = text_preprocessor.fit_transform(raw_data)
        self.y = self._lb.fit_transform(labels)

        del raw_data
        del labels

        self.V = text_preprocessor.named_steps['idx'].vocab
        self.idx2w = text_preprocessor.named_steps['idx'].idx2w
        self.w2idx = text_preprocessor.named_steps['idx'].w2idx

    def _process_dir(self, directory):
        X = []
        y = []

        for file in os.listdir(directory):
            if os.path.isdir(directory + '/' + file):
                subdir_texts, labels = self._process_dir(
                    directory + '/' + file)
                for text in subdir_texts:
                    X.append(text)
                for label in labels:
                    y.append(label)
            elif os.path.isfile(directory + '/' + file):
                sents = open(
                    directory + '/' + file,
                    'rb').read().decode(
                    'utf-8',
                    'ignore').splitlines()
                for sent in sents:
                    X.append(sent)
                    y.append(directory)
            else:
                continue

        return X, y

    def _process_file(self, file):
        if file[-3:] == 'csv':
            X, y = self._process_csv(file)
        else:
            pass

        return X, y

    def _process_csv(self, file):
        df = pd.read_csv(file, header=0, delimiter='|')
        df = df[df['phrase'].notnull()]
        df = df[df['label'].notnull()]

        X = df['phrase'].values
        y = df['label'].values

        return X, y

    def _process_list(self, l):
        '''
        Process list of lists, where each l[0] = str and l[1] = label
        Only to be used for testing
        :param l:
        '''
        X = [x[0] for x in l]
        y = [y[1] for y in l]
        return X, y

    def batch_data(self, X, y, minibatch_size=32, shuffle=True):
        l = len(X)
        if minibatch_size > l:
            raise AttributeError(
                "Error, {} must be smaller than {}".format(
                    minibatch_size, l))

        if shuffle:
            p = np.random.permutation(l)
            X, y = X[p], y[p]
        for i in range(0, l, minibatch_size):
            yield X[i:i + minibatch_size], y[i:i + minibatch_size]

    def __len__(self):
        return len(self.V)

    def __str__(self):
        return "Dataset built off of: {}\nVocab size: {}".format(
            self.dir, len(self.V)
        )
