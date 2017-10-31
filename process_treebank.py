import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datafile', help='directory containing SSTB', required=True, dest='datafile')
    return vars(parser.parse_args())

def parse_sstb(dir, binary=False):
    '''
    Hardcoded solution to process SSTB due to its unique formatting
    :param dir: SSTB directory
    :return: dictionary containing phrase/label combinations
    '''
    dict = open(dir + '/dictionary.txt', 'r').read()
    labels = open(dir + '/sentiment_labels.txt', 'r').read()

    tuples = [sub.split('|') for sub in dict.split('\n')]
    phrase_id = {x[0]: int(x[1]) for x in tuples if len(x) > 1}
    tuples2 = [sub.split('|') for sub in labels.split('\n')][1:]
    id_label = {int(x[0]): float(x[1]) for x in tuples2 if len(x) > 1}

    phrase_label = {x: transform_label(id_label[phrase_id[x]]) for x in phrase_id.keys()}

    # Binarize
    if binary:
        phrase_label = {x:binarize(y) for (x, y) in phrase_label.items() if y != 2}

    return phrase_label

def transform_label(label):
    # Hardcode as per specs from SSTB
    if label <= 0.2:
        return 0
    elif label > 0.2 and label <= 0.4:
        return 1
    elif label > 0.4 and label <= 0.6:
        return 2
    elif label > 0.6 and label <= 0.8:
        return 3
    elif label > 0.8 and label <= 1:
        return 4

def binarize(y):
    if y > 2:
        return 1
    elif y < 2:
        return 0

if __name__ == '__main__':
    args = parse_args()
    print('loading SSTB...')
    dict = parse_sstb(args['datafile'])
    print(dict.items()[:10])
    print('Outputting reformatted version to {}'.format(args['datafile'] + '/formatted_sstb.csv'))
    with open(args['datafile'] + '/formatted_sstb.csv', 'w') as writefile:
        writefile.write('phrase|label')
        writefile.write('\n')
        for (x, y) in dict.items():
            writefile.write(x + '|' + str(y))
            writefile.write('\n')