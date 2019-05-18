import argparse
import os
import numpy as np

# python data.py --gram SL --k 2 --n 10k

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('--gram', type=str, default='SL', help = 'SL:strick local, SP:strict piecewise')
parser.add_argument('--k', type=str, default='2', help = '2, 4 or 8')
parser.add_argument('--n', type=str, default='1k', help = 'number of strings')
parser.add_argument('--bsz', type=int, default= 32, help = 'seed for shuffling data')
parser.add_argument('--seed', type=int, default= 123, help = 'seed for shuffling data')

args = parser.parse_args()
np.random.seed(args.seed)

def get_shuffled_ids(data, bsize):
    sorted_ids = np.argsort([len(l)+np.random.uniform(-1.0,1.0) for l in data])
    blocked_sorted_ids = [ sorted_ids[i:i+bsize] for i in xrange(0,len(data),bsize) ]
    np.random.shuffle( blocked_sorted_ids )
    return blocked_sorted_ids

class Grammar_Corpus(object):
    def __init__(self, path):
        data = {'tr': {}, 'te1': {}, 'te2': {}}
        label = {'tr': {}, 'te1': {}, 'te2': {}}
        pat_name = {'tr': 'Training', 'te1': 'Test1', 'te2': 'Test2'}
        self.alphabet = None

        for key in pat_name.keys():
            file_name = ''.join((path, pat_name[key], '.txt'))
            pd = np.array([l.strip().split() for l in open(file_name)])
            data[key] = pd[:, 0]
            label[key] = np.array([1 if l == 'TRUE' else 0 for l in list(pd[:, 1])], dtype='int32')

        self.shuffled_ids = get_shuffled_ids(data['tr'], args.bsz)

        self.train, self.train_mask = self.tokenize(data['tr'])
        self.test1, self.test1_mask = self.tokenize(data['te1'])
        self.test2, self.test2_mask = self.tokenize(data['te2'])

        self.train_label = label['tr']
        self.test1_label = label['te1']
        self.test2_label = label['te2']

    def tokenize(self, strings):
        if self.alphabet is None:
            self.alphabet = sorted(set([a for l in strings for a in l]))
            self.dictionary = {a: i for i, a in enumerate(self.alphabet)}
        else:
            alphabet = sorted(set([a for l in strings for a in l]))
            assert alphabet == self.alphabet


        lens = [len(s) for s in strings]
        w = max(lens)
        h = len(strings)
        mask = np.zeros((h, w), dtype='int32')
        data = np.zeros((h, w), dtype='int32')
        for i, s in enumerate(strings):
            mask[i, :len(s)] = 1
            data[i, :len(s)] = [self.dictionary[a] for a in s]

        return data, mask

data_folder = './data/' + args.gram + '/' + args.gram + args.k + '/' + args.n + '/'
data_file = data_folder + args.gram + args.k + '-' + args.n + '-'
train_val_test_file = ''.join((data_folder, 'train_val_test_data.npz'))

corpus = Grammar_Corpus(data_file)

idx_list = np.arange(corpus.train.shape[0], dtype='int')
np.random.shuffle(idx_list)
corpus.train = corpus.train[idx_list]
corpus.train_mask = corpus.train_mask[idx_list]
corpus.train_label = corpus.train_label[idx_list]

np.savez(train_val_test_file, batch_idx = corpus.shuffled_ids, alphabet=corpus.alphabet,
         train_x=corpus.train, train_m=corpus.train_mask, train_y=corpus.train_label,
         test1_x=corpus.test1, test1_m=corpus.test1_mask, test1_y=corpus.test1_label,
         test2_x=corpus.test2, test2_m=corpus.test2_mask, test2_y=corpus.test2_label)


