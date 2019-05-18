import argparse
import numpy as np

# python data.py --number 61 --split 0.7

parser = argparse.ArgumentParser(description='Preprocess data')
parser.add_argument('--number', type=str, default='61', help = 'RNA data')
parser.add_argument('--split', type=float, default=0.7, help = 'train ratio')
parser.add_argument('--seed', type=int, default= 123, help = 'seed for shuffling data')

args = parser.parse_args()
np.random.seed(args.seed)

def get_shuffled_ids(data, bsize):
    sorted_ids = np.argsort([len(l)+np.random.uniform(-1.0,1.0) for l in data])
    blocked_sorted_ids = [ sorted_ids[i:i+bsize] for i in xrange(0,len(data),bsize) ]
    np.random.shuffle( blocked_sorted_ids )
    return blocked_sorted_ids

class STAMINA_Corpus(object):
    def __init__(self, filename, split):

        self.data = {'tr_data':[], 'tr_mask':[], 'tr_label':[],
                     'te_data':[], 'te_mask':[], 'te_label':[],}
        self.alphabet = None
        self.max_len = 0
        self.split = split

        data = []
        label = []
        data_raw = [l for l in open(filename)]
        for seq in data_raw:
            label += [1 if seq[0] == '+' else 0]
            data.append([int(l) for l in seq[1:].split()])

        self.tokenize(data, label)

    def tokenize(self, data, label):
        if self.alphabet is None:
            self.alphabet = sorted(set([a for l in data for a in l]))
            self.dictionary = {a: i for i, a in enumerate(self.alphabet)}

        lens = [len(l) for l in data]
        self.max_len = max(lens)
        self.n = len(data)

        data_all = np.zeros((self.n, self.max_len), dtype='int32')
        mask = np.zeros((self.n, self.max_len), dtype='int32')
        label = np.array(label, dtype='int16')

        for i, s in enumerate(data):
            mask[i, :len(s)] = 1
            data_all[i, :len(s)] = [self.dictionary[a] for a in s]

        pos_id = np.where(label == 1)[0]
        neg_id = np.where(label == 0)[0]

        data_pos = data_all[pos_id,:]
        data_neg = data_all[neg_id,:]
        mask_pos = mask[pos_id,:]
        mask_neg = mask[neg_id,:]

        self.n_pos = pos_id.shape[0]
        self.n_neg = neg_id.shape[0]

        n_pos_train = int(data_pos.shape[0] * self.split)
        n_neg_train = int(data_neg.shape[0] * self.split)

        def split_train_test(data, mask, n_train):
            idx_list = np.arange(data.shape[0], dtype='int')
            np.random.shuffle(idx_list)
            data = data[idx_list]
            mask = mask[idx_list]
            return data[:n_train], mask[:n_train], data[n_train:], mask[n_train:]

        def prepare_x_m_y(x_pos, m_pos, x_neg, m_neg):
            x = np.concatenate((x_pos, x_neg), axis=0)
            y = np.concatenate((np.ones((x_pos.shape[0],), dtype='int32'),
                                np.zeros((x_neg.shape[0],), dtype='int32')), axis=0)
            m = np.concatenate((m_pos, m_neg), axis=0)

            idx_list = np.arange(x.shape[0], dtype='int')
            np.random.shuffle(idx_list)
            x = x[idx_list]
            y = y[idx_list]
            m = m[idx_list]

            return x, m, y

        train_pos, train_mask_pos, test_pos, test_mask_pos = split_train_test(data_pos, mask_pos, n_pos_train)
        train_neg, train_mask_neg, test_neg, test_mask_neg = split_train_test(data_neg, mask_neg, n_neg_train)

        self.data['tr_data'], self.data['tr_mask'], self.data['tr_label'] = \
            prepare_x_m_y(train_pos, train_mask_pos, train_neg, train_mask_neg)
        self.data['te_data'], self.data['te_mask'], self.data['te_label'] = \
            prepare_x_m_y(test_pos, test_mask_pos, test_neg, test_mask_neg)

        self.n_pos_train = n_pos_train
        self.n_neg_train = n_neg_train


data_file = './data/STAMINA/' + args.number + '_training.txt'
train_val_test_file = ''.join(('./data/STAMINA/' + args.number, '_train_val_test_data.npz'))

corpus = STAMINA_Corpus(data_file, args.split)
print('| All {:5d} | Pos {:5d} | Neg {:5d} | Pos Train {:5d} | Neg Train {:5d} |'.format(
    corpus.n, corpus.n_pos, corpus.n_neg, corpus.n_pos_train, corpus.n_neg_train))

np.savez(train_val_test_file, alphabet=corpus.alphabet,
         train_x=corpus.data['tr_data'], train_m=corpus.data['tr_mask'], train_y=corpus.data['tr_label'],
         test_x=corpus.data['te_data'], test_m=corpus.data['te_mask'], test_y=corpus.data['te_label'])


