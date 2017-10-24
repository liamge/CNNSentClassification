mport os, warnings
import torch.utils.data as data
from sklearn.feature_extraction.text import CountVectorizer
from spacy.en import English

def sentence_tokenize(corpus):
    '''
    Spacy sentence tokenizer to split corpus into unique sentences
    :param corpus: object of type string or list containing exactly 1 string to be tokenized
    :return: list of strings of sentences
    '''
    # Handling different types of input
    if type(corpus) == str:
        nlp = English()
        doc = nlp(corpus)
        return [sent.string.strip() + ['<EOS>'] for sent in doc.sents]
    elif type(corpus) == list and len(corpus) == 1:
        nlp = English()
        doc = nlp(corpus[0])
        return [sent.string.strip() for sent in doc.sents]
    else:
        raise TypeError("Error: corpus parameter was of type: {}\n"
                        "Corpus needs to be either a singleton array of a string or a string")

class TypeError(Exception):
    '''
    Unnecessary but educational custom error message to use for sentence_tokenize
    '''
    def __init__(self, value):
        self.parameter = value
    def __str__(self):
        return repr(self.parameter)

class DataLoader:
    '''
    Basic data loader to load in a directory or file of data as well as useful helper functions
    '''
    def __init__(self, directory, batch_size, shuffle=True, num_workers=2):
        self.dir, self.batch_size = directory, batch_size

        # Exception handling for various input formats
        if os.path.isdir(directory):
            self._raw_data = self._process_dir(directory)
        elif os.path.isfile(directory):
            self._raw_data = self._process_file(directory)
        assert self._raw_data is not None, \
            "Error: Something's gone wrong, please check the contents of {}\n" \
            "The format must be either a directory of files, a directory of directories," \
            "or a file.".format(directory)

        # Count tensorization
        self._count_tensor = CountVectorizer().fit_transform(self._raw_data)
        self.V = self._count_tensor.vocabulary_

        # Use torch's dataloader
        self.data = data.DataLoader(self._count_tensor, batch_size, shuffle, num_workers)

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

    def __len__(self):
        warnings.warn("usage of len on DataLoader returns length of vocabulary,"
                      "for number of batches please use len(DataLoader.data),"
                      "for length of the raw dataset use len(DataLoader._raw_data)", Warning)
        return len(self.V)

    def __str__(self):
        return "Dataset built off of: {}\nBatch size:  {}\nNumber of batches: {}".format(
            self.dir, self.batch_size, len(self.data)
        )
