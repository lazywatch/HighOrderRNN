from collections import OrderedDict
import numpy as np
from sklearn import metrics

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def read_model_coef(model):
    #print('Reading model ...')
    model_params = unzip(model.tparams)
    return model_params

def update_model(model, model_predict):
    model_params = read_model_coef(model)

    for kk, vv in model_predict.tparams.items():
        if kk not in model_params:
            raise Warning('%s is not in the archive' % kk)
        model_predict.tparams[kk].set_value(model_params[kk])

    #model_predict.build_model()

    return model_predict

def load_params(path, path_hinit, params):
    hinit = np.load(path_hinit)['hinit']

    pp = np.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params, hinit


def save_hinit(hinit, path_hinit):
    print('Saving hinit ... \n')
    np.savez(path_hinit, hinit=hinit)

def split_train_test(data, mask, n_train, n_val):
    #idx_list = np.arange(data.shape[0])
    #data = data[idx_list]
    #mask = mask[idx_list]
    train = data[:n_train]
    train_mask = mask[:n_train]

    val = data[n_train:n_train+n_val]
    val_mask = mask[n_train:n_train+n_val]

    test = data[n_train+n_val:]
    test_mask = mask[n_train+n_val:]

    return train, train_mask, val, val_mask, test, test_mask


def prepare_x_m_y(pos_x, pos_m, neg_x, neg_m):
    x = np.concatenate((pos_x, neg_x), axis=0)
    y = np.concatenate((np.ones((pos_x.shape[0],), dtype='int32'),
                        np.zeros((neg_x.shape[0],), dtype='int32')), axis=0)
    m = np.concatenate((pos_m, neg_m), axis=0)

    idx_list = np.arange(x.shape[0], dtype='int')
    np.random.shuffle(idx_list)
    x = x[idx_list]
    y = y[idx_list]
    m = m[idx_list]

    return x, y, m


def load_data(data_dir, train_test_split=0.15, val_ratio=0.05):#, seed=123):
    #np.random.seed(seed)
    npzfile = np.load(data_dir)
    pos_set = npzfile['pos_set']
    pos_mask = npzfile['pos_mask']
    neg_set = npzfile['neg_set']
    neg_mask = npzfile['neg_mask']

    n_pos_train = int(pos_set.shape[0] * train_test_split)
    n_neg_train = int(neg_set.shape[0] * train_test_split)
    n_pos_val = int(pos_set.shape[0] * val_ratio)
    n_neg_val = int(neg_set.shape[0] * val_ratio)


    idx_list = np.arange(pos_set.shape[0], dtype='int')
    np.random.shuffle(idx_list)
    pos_set = pos_set[idx_list]
    pos_mask = pos_mask[idx_list]
    pos_train, pos_train_mask, pos_val, pos_val_mask, pos_test, pos_test_mask = \
        split_train_test(pos_set,pos_mask,n_pos_train,n_pos_val)

    idx_list = np.arange(neg_set.shape[0], dtype='int')
    np.random.shuffle(idx_list)
    neg_set = neg_set[idx_list]
    neg_mask = neg_mask[idx_list]
    neg_train, neg_train_mask, neg_val, neg_val_mask, neg_test, neg_test_mask = \
        split_train_test(neg_set, neg_mask, n_neg_train, n_neg_val)

    train_x, train_y, train_m = prepare_x_m_y(pos_x=pos_train, pos_m=pos_train_mask,
                                              neg_x=neg_train, neg_m=neg_train_mask)
    val_x, val_y, val_m = prepare_x_m_y(pos_x=pos_val, pos_m=pos_val_mask,
                                        neg_x=neg_val, neg_m=neg_val_mask)
    test_x, test_y, test_m = prepare_x_m_y(pos_x=pos_test, pos_m=pos_test_mask,
                                           neg_x=neg_test, neg_m=neg_test_mask)

    return train_x, train_m, train_y, val_x, val_m, val_y, test_x, test_m, test_y

def get_minibatches_idx(n, minibatch_size, shuffle=True, data_type = 'int'):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = np.arange(n, dtype=data_type)

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        remain_idx = idx_list[minibatch_start:]
        assert remain_idx.shape[0] == (n%minibatch_size)

        if idx_list.shape[0] > minibatch_size:
            remain_idx = np.concatenate((remain_idx, idx_list[:(minibatch_size - remain_idx.shape[0])]))
        else:
            append_cnt = 0
            while remain_idx.shape[0] < minibatch_size:
                try:
                    remain_idx = np.concatenate((remain_idx, [idx_list[append_cnt%idx_list.shape[0]]]))
                    append_cnt += 1
                except IndexError:
                    print(idx_list)
                    print(append_cnt)
                    print(remain_idx)


        try:
            assert remain_idx.shape[0] == minibatch_size
        except AssertionError:
            print('Assertion Error: maybe total number of samples is smaller than batch size')
        minibatches.append(remain_idx)

    return zip(range(len(minibatches)), minibatches)

def perf_measure(y_true, y_pred, ep = 0.5, use_self=False):

    if use_self:
        pos_label = 1.0
        neg_label = 0.0

        pos_id = np.where(y_true == pos_label)[0]
        aa = np.where(y_pred[pos_id]>=1.0-ep)[0]
        TP = float(aa.shape[0])

        aa = np.where(y_pred[pos_id]<1.0-ep)[0]
        FN = float(aa.shape[0])

        '''
        if not (FN == 0.0):
        #    print('FP id:')
            FN_id = pos_id[aa]
        #    print(FP_id)
        else:
            FN_id = None
        '''

        neg_id = np.where(y_true == neg_label)[0]

        aa = np.where(y_pred[neg_id] < ep)[0]
        TN = float(aa.shape[0])


        aa = np.where(y_pred[neg_id]>=ep)[0]
        FP = float(aa.shape[0])

        #print ("TP: %s FP: %s TN: %s FN: %s" % (TP, FP, TN, FN))

        '''
        if not (FP == 0.0):
        #    print('FP id:')
            FP_id = neg_id[aa]
        #    print(FP_id)
        else:
            FP_id = None
        '''

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        #f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        c_minus = TN / (TN + FP + 1e-9)
        bcr = (2 * c_minus * recall) / (c_minus + recall + 1e-9)
        return (precision, recall, accuracy, bcr)
    else:
        y_pred_int = np.ones_like(y_pred, dtype='int32')
        aa = np.where(y_pred < 1.0-ep)[0]
        y_pred_int[aa] = 0

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred_int)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred_int)
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_int)
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred_int)

        return(precision, recall, accuracy, f1)

def perf_measure_bcr(y_true, y_pred, ep = 0.5, use_self=False):

    if use_self:
        pos_label = 1.0
        neg_label = 0.0

        pos_id = np.where(y_true == pos_label)[0]
        aa = np.where(y_pred[pos_id]>=1.0-ep)[0]
        TP = float(aa.shape[0])

        aa = np.where(y_pred[pos_id]<1.0-ep)[0]
        FN = float(aa.shape[0])

        '''
        if not (FN == 0.0):
        #    print('FP id:')
            FN_id = pos_id[aa]
        #    print(FP_id)
        else:
            FN_id = None
        '''

        neg_id = np.where(y_true == neg_label)[0]

        aa = np.where(y_pred[neg_id] < ep)[0]
        TN = float(aa.shape[0])


        aa = np.where(y_pred[neg_id]>=ep)[0]
        FP = float(aa.shape[0])

        #print ("TP: %s FP: %s TN: %s FN: %s" % (TP, FP, TN, FN))

        '''
        if not (FP == 0.0):
        #    print('FP id:')
            FP_id = neg_id[aa]
        #    print(FP_id)
        else:
            FP_id = None
        '''

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        #f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        c_minus = TN / (TN + FP + 1e-9)
        bcr = (2 * c_minus * recall) / (c_minus + recall + 1e-9)
        return (precision, recall, accuracy, bcr)
    else:
        y_pred_int = np.ones_like(y_pred, dtype='int32')
        aa = np.where(y_pred < 1.0-ep)[0]
        y_pred_int[aa] = 0

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred_int)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred_int)
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred_int)
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred_int)

        return(precision, recall, accuracy, f1)

def dfa_perf_measure(y_true, y_pred, ep = 0.5, use_self=False):

    if use_self:
        pos_label = 1.0
        neg_label = 0.0

        pos_id = np.where(y_true == pos_label)[0]
        aa = np.where(y_pred[pos_id]>=1.0-ep)[0]
        TP = float(aa.shape[0])

        aa = np.where(y_pred[pos_id]<1.0-ep)[0]
        FN = float(aa.shape[0])

        neg_id = np.where(y_true == neg_label)[0]
        aa = np.where(y_pred[neg_id]>=ep)[0]
        FP =float(aa.shape[0])

        aa = np.where(y_pred[neg_id]<ep)[0]
        TN = float(aa.shape[0])

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1 = (2 * precision * recall) / (precision + recall+1e-9)

        #print ("TP: %s FP: %s TN: %s FN: %s" % (TP, FP, TN, FN))
        return (TP, FP, TN, FN, precision, recall, accuracy, f1)
    else:
        #y_pred_int = np.ones_like(y_pred, dtype='float64')
        #aa = np.where(y_pred < 1.0-ep)[0]
        #y_pred_int[aa] = 0

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred)
        accuracy = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = metrics.f1_score(y_true=y_true, y_pred=y_pred)

        return(precision, recall, accuracy, f1)



def param_count(model, I, H):
    if model == 'UNI': # 4,28, 4060
        return I*H*H + I*H + H*H + H
    elif model == 'O2': # 4,31, 3875
        return I*H*H + H
    elif model == 'M': # 4,42, 3906
        return 2*I*H + 2*H*H + H
    elif model == 'MI': #4,60, 4080
        return I*H + H*H + 4*H
    elif model == 'SRN': #4,62, 4154
        return I*H + H*H + H
    elif model == 'LSTM': # I,H = 4,30, 4080
        return 4*I*H + 4*H*H
    elif model == 'GRU': # 4,34, 3876
        return 3*H*I + 3*H*H