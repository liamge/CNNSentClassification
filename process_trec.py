import argparse, os
from sklearn.preprocessing import LabelEncoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', help='directory containing TREC', required=True, dest='datafile')
    return vars(parser.parse_args())

def process_dir(directory):
    full_labels = []
    full_text = []

    for filename in os.listdir(directory):
        file = open(directory + filename, 'rb').read().decode('utf-8', 'ignore')
        tmp = [(x.split()[0], ' '.join(x.split()[1:])) for x in file.split('\n')[:-1]]
        labels = [x[0] for x in tmp]
        full_labels.append(labels)
        text = [x[1] for x in tmp]
        full_text.append(text)

    # Flatten list of lists
    full_labels = [item for sublist in full_labels for item in sublist]
    full_text = [item for sublist in full_text for item in sublist]

    le = LabelEncoder()
    full_labels = le.fit_transform(full_labels)

    return dict(zip(full_text, full_labels))

if __name__ == '__main__':
    args = parse_args()
    print('Loading TREC training data...')
    train_data = process_dir(args['datafile']+'/train/')
    print('Loading TREC testing data...')
    test_data = process_dir(args['datafile'] + '/test/')

    print('Writing data to {}'.format(args['datafile']))

    with open(args['datafile'] + '/train/processed.csv', 'w') as writefile:
        writefile.write('phrase|label')
        writefile.write('\n')
        for (x, y) in train_data.items():
            writefile.write(x + '|' + str(y))
            writefile.write('\n')

    with open(args['datafile'] + '/test/processed.csv', 'w') as writefile:
        writefile.write('phrase|label')
        writefile.write('\n')
        for (x, y) in train_data.items():
            writefile.write(x + '|' + str(y))
            writefile.write('\n')