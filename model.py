from collections import OrderedDict
import math
import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.gradient import grad_clip

theano.config.optimizer = 'fast_compile'
#theano.config.exception_verbosity = 'high'
#theano.config.compute_test_value = 'warn'

config.floatX = 'float32'



def numpy_floatX(data):
    return np.asarray(data, dtype=config.floatX)

def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent
    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.
    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')
    return f_grad_shared, f_update

def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')#, profile=True)
    #f_grad_shared.profile.summary()

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update

def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[0][0].shape[0]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(tensor, mode='fan_in'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, param=None):
    r"""Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`\frac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{\frac{2}{1 + \text{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu')
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std #- 0.5# Calculate uniform bounds from standard deviation
    return np.random.uniform(low=-bound, high=bound, size=tensor.shape).astype(config.floatX)


def xavier_uniform_(tensor, gain=1):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    scale = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(low=-scale, high=scale, size=tensor.shape).astype(config.floatX)

def glorot_uniform_T(shape):
    fan_in, fan_h, fan_out = shape[0], shape[1], shape[2]
    scale = np.sqrt(6. / (fan_in + fan_h + fan_out))
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(config.floatX)

def glorot_uniform(shape):
    fan_in, fan_out = shape[0], shape[1]
    scale = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(config.floatX)

def ortho_weight(shape):
    ndim1, ndim2 = shape
    assert ndim1 == ndim2
    W = np.random.randn(ndim1, ndim2)
    u, s, v = np.linalg.svd(W)
    return u.astype(config.floatX)

def ortho_weight_T(shape):
    indim, hdim1, hdim2 = shape
    assert hdim1 == hdim2
    W = np.zeros((indim, hdim1, hdim2),dtype=config.floatX)
    for i in range(indim):
        W[i,:] = ortho_weight([hdim1,hdim2])
    return W

def _p(pp, name):
    return '%s_%s' % (pp, name)

class uni_layer():
    def __init__(self, rnn_type, ninp, nhid, nonlinearity):
        self.params = OrderedDict()
        #W = kaiming_uniform_(np.zeros([ninp, nhid, nhid]), nonlinearity=nonlinearity)
        #U = kaiming_uniform_(np.zeros([ninp, nhid]), nonlinearity=nonlinearity)
        #V = kaiming_uniform_(np.zeros([nhid, nhid]), nonlinearity=nonlinearity)
        #fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([ninp, nhid]))
        #bound = 1 / math.sqrt(fan_in)# - 0.5
        #B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        #W = ortho_weight_T([ninp, nhid, nhid])
        #U = np.zeros([ninp, nhid], dtype=config.floatX)
        #V = np.zeros([nhid, nhid], dtype=config.floatX)
        #B = 2 * np.ones(nhid, dtype=config.floatX)

        bound = 0.02 + 0.5
        W = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid, nhid])).astype(config.floatX)
        U = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        V = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        self.params[_p(rnn_type, 'W')] = W
        self.params[_p(rnn_type, 'U')] = U
        self.params[_p(rnn_type, 'V')] = V
        self.params[_p(rnn_type, 'B')] = B

        self.nonlinearity = nonlinearity
        self.prefix = rnn_type

        def sig_cell(x_, m_, h_, w, u, v, b):
            wx = T.tensordot(x_, w, [[1], [0]])
            wxh = T.batched_dot(wx, h_) + b
            h_pre = wxh + T.dot(x_, u) + T.dot(h_, v)

            h = T.nnet.sigmoid(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def tanh_cell(x_, m_, h_, w, u, v, b):
            wx = T.tensordot(x_, w, [[1], [0]])
            wxh = T.batched_dot(wx, h_) + b
            h_pre = wxh + T.dot(x_, u) + T.dot(h_, v)

            h = T.tanh(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def relu_cell(x_, m_, h_, w, u, v, b):
            wx = T.tensordot(x_, w, [[1], [0]])
            wxh = T.batched_dot(wx, h_) + b
            h_pre = wxh + T.dot(x_, u) + T.dot(h_, v)

            h = T.nnet.relu(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        if self.nonlinearity == 'sigmoid':
            self.rnn_cell = sig_cell
        elif self.nonlinearity == 'tanh':
            self.rnn_cell = tanh_cell
        elif self.nonlinearity == 'relu':
            self.rnn_cell = relu_cell

    def inner(self, x, m, h_, tparams, nsteps):

        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'W')],
                                                tparams[_p(self.prefix, 'U')],
                                                tparams[_p(self.prefix, 'V')],
                                                tparams[_p(self.prefix, 'B')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates


class o2_layer():
    def __init__(self, rnn_type, ninp, nhid, nonlinearity):
        self.params = OrderedDict()
        #W = kaiming_uniform_(np.zeros([ninp, nhid, nhid]), nonlinearity=nonlinearity)
        #fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([ninp, nhid]))
        #bound = 1 / math.sqrt(fan_in)
        #B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        #W = ortho_weight_T([ninp, nhid, nhid])
        # B = 2 * np.ones(nhid, dtype=config.floatX)

        bound = 0.02  # - 0.5
        W = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid, nhid])).astype(config.floatX)
        B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        self.params[_p(rnn_type, 'W')] = W
        self.params[_p(rnn_type, 'B')] = B

        self.nonlinearity = nonlinearity
        self.prefix = rnn_type

        def sig_cell(x_, m_, h_, w, b):
            wh = T.tensordot(x_, w, [[1], [0]])
            h_pre = T.batched_dot(wh, h_) + b
            h = T.nnet.sigmoid(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def tanh_cell(x_, m_, h_, w, b):
            wh = T.tensordot(x_, w, [[1], [0]])
            h_pre = T.batched_dot(wh, h_) + b
            h = T.tanh(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def relu_cell(x_, m_, h_, w, b):
            wh = T.tensordot(x_, w, [[1], [0]])
            h_pre = T.batched_dot(wh, h_) + b
            h = T.nnet.relu(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        if self.nonlinearity == 'sigmoid':
            self.rnn_cell = sig_cell
        elif self.nonlinearity == 'tanh':
            self.rnn_cell = tanh_cell
        elif self.nonlinearity == 'relu':
            self.rnn_cell = relu_cell

    def inner(self, x, m, h_, tparams, nsteps):
        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'W')],
                                                tparams[_p(self.prefix, 'B')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates

class m_layer():
    def __init__(self, rnn_type, ninp, nhid, nonlinearity='tanh'):
        self.nf = 3
        self.params = OrderedDict()
        #fx = kaiming_uniform_(np.zeros([ninp, self.nf]), nonlinearity=nonlinearity)
        #fh = kaiming_uniform_(np.zeros([nhid, self.nf]), nonlinearity=nonlinearity)
        #hf = kaiming_uniform_(np.zeros([self.nf, nhid]), nonlinearity=nonlinearity)
        #hx = kaiming_uniform_(np.zeros([ninp, nhid]), nonlinearity=nonlinearity)

        #fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([ninp, nhid]))
        #bound = 1 / math.sqrt(fan_in)
        #B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        #fx = glorot_uniform([ninp, self.nf])
        #fh = ortho_weight([nhid, self.nf])
        #hf = ortho_weight([nhid, nhid])
        #hx = glorot_uniform([ninp, nhid])
        # B = 2 * np.ones(nhid, dtype=config.floatX)

        bound = 0.02  # - 0.5
        fx = np.random.uniform(low=-bound, high=bound, size=([ninp, self.nf])).astype(config.floatX)
        fh = np.random.uniform(low=-bound, high=bound, size=([nhid, self.nf])).astype(config.floatX)
        hf = np.random.uniform(low=-bound, high=bound, size=([self.nf, nhid])).astype(config.floatX)
        hx = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        self.params[_p(rnn_type, 'fx')] = fx
        self.params[_p(rnn_type, 'fh')] = fh
        self.params[_p(rnn_type, 'hf')] = hf
        self.params[_p(rnn_type, 'hx')] = hx
        self.params[_p(rnn_type, 'B')] = B

        self.nonlinearity = nonlinearity
        self.prefix = rnn_type

        def batch_diag(x_b, w):
            return T.nlinalg.diag(T.dot(x_b, w))

        def sig_cell(x_, m_, h_, fx, fh, hf, hx, b):
            nbatch = x_.shape[0]
            tmp0, _ = theano.scan(batch_diag, sequences=[x_], non_sequences=fx, outputs_info=None, n_steps=nbatch)
            tmp1 = T.dot(h_, fh)
            ft = T.batched_tensordot(tmp0, tmp1, [[2], [1]])
            h_pre = T.nnet.sigmoid(T.dot(ft, hf) + T.dot(x_, hx) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        def tanh_cell(x_, m_, h_, fx, fh, hf, hx, b):
            nbatch = x_.shape[0]
            tmp0, _ = theano.scan(batch_diag, sequences=[x_], non_sequences=fx, outputs_info=None, n_steps=nbatch)
            tmp1 = T.dot(h_, fh)
            ft = T.batched_tensordot(tmp0, tmp1, [[2], [1]])
            h_pre = T.tanh(T.dot(ft, hf) + T.dot(x_, hx) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        def relu_cell(x_, m_, h_, fx, fh, hf, hx, b):
            nbatch = x_.shape[0]
            tmp0, _ = theano.scan(batch_diag, sequences=[x_], non_sequences=fx, outputs_info=None, n_steps=nbatch)
            tmp1 = T.dot(h_, fh)
            ft = T.batched_tensordot(tmp0, tmp1, [[2], [1]])
            h_pre = T.nnet.relu(T.dot(ft, hf) + T.dot(x_, hx) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        if self.nonlinearity == 'sigmoid':
            self.rnn_cell = sig_cell
        elif self.nonlinearity == 'tanh':
            self.rnn_cell = tanh_cell
        elif self.nonlinearity == 'relu':
            self.rnn_cell = relu_cell

    def inner(self, x, m, h_, tparams, nsteps):
        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'fx')],
                                                tparams[_p(self.prefix, 'fh')],
                                                tparams[_p(self.prefix, 'hf')],
                                                tparams[_p(self.prefix, 'hx')],
                                                tparams[_p(self.prefix, 'B')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates

class mi_layer():
    def __init__(self, rnn_type, ninp, nhid, nonlinearity='tanh'):
        self.params = OrderedDict()
        #U = kaiming_uniform_(np.zeros([ninp, nhid]), nonlinearity=nonlinearity)
        #V = kaiming_uniform_(np.zeros([nhid, nhid]), nonlinearity=nonlinearity)
        #fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([ninp, nhid]))
        #bound = 1 / math.sqrt(fan_in)
        #B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)
        alpha = 2 * np.ones(nhid, dtype=config.floatX)
        beta1 = 0.5 * np.ones(nhid, dtype=config.floatX)
        beta2 = 0.5 * np.ones(nhid, dtype=config.floatX)

        #U = glorot_uniform([ninp, nhid])
        #V = ortho_weight([nhid, nhid])

        bound = 0.02  # - 0.5
        U = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        V = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        B = 2 * np.zeros(nhid, dtype=config.floatX)


        self.params[_p(rnn_type, 'U')] = U
        self.params[_p(rnn_type, 'V')] = V
        self.params[_p(rnn_type, 'B')] = B
        self.params[_p(rnn_type, 'alpha')] = alpha
        self.params[_p(rnn_type, 'beta1')] = beta1
        self.params[_p(rnn_type, 'beta2')] = beta2

        self.nonlinearity = nonlinearity
        self.prefix = rnn_type

        def sig_cell(x_, m_, h_, u, v, b, alpha, beta1, beta2):
            h_pre = T.nnet.sigmoid(alpha * T.dot(x_, u) * T.dot(h_, v) +
                                   beta1 * T.dot(h_, v) + beta2 * T.dot(x_, u) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        def tanh_cell(x_, m_, h_, u, v, b, alpha, beta1, beta2):
            h_pre = T.tanh(alpha * T.dot(x_, u) * T.dot(h_, v) +
                           beta1 * T.dot(h_, v) + beta2 * T.dot(x_, u) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        def relu_cell(x_, m_, h_, u, v, b, alpha, beta1, beta2):
            h_pre = T.nnet.relu(alpha * T.dot(x_, u) * T.dot(h_, v) +
                                beta1 * T.dot(h_, v) + beta2 * T.dot(x_, u) + b)
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        if self.nonlinearity == 'sigmoid':
            self.rnn_cell = sig_cell
        elif self.nonlinearity == 'tanh':
            self.rnn_cell = tanh_cell
        elif self.nonlinearity == 'relu':
            self.rnn_cell = relu_cell

    def inner(self, x, m, h_, tparams, nsteps):
        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'U')],
                                                tparams[_p(self.prefix, 'V')],
                                                tparams[_p(self.prefix, 'B')],
                                                tparams[_p(self.prefix, 'alpha')],
                                                tparams[_p(self.prefix, 'beta1')],
                                                tparams[_p(self.prefix, 'beta2')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates


class srn_layer():
    def __init__(self, rnn_type, ninp, nhid, nonlinearity):
        self.params = OrderedDict()
        #U = kaiming_uniform_(np.zeros([ninp, nhid]), nonlinearity=nonlinearity)
        #V = kaiming_uniform_(np.zeros([nhid, nhid]), nonlinearity=nonlinearity)
        #fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([ninp, nhid]))
        #bound = 1 / math.sqrt(fan_in)
        #B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        #U = glorot_uniform([ninp, nhid])
        #V = ortho_weight([nhid, nhid])

        bound = 0.02  # - 0.5
        U = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        V = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        B = np.random.uniform(low=-bound, high=bound, size=nhid).astype(config.floatX)

        self.params[_p(rnn_type, 'U')] = U
        self.params[_p(rnn_type, 'V')] = V
        self.params[_p(rnn_type, 'B')] = B

        self.nonlinearity = nonlinearity
        self.prefix = rnn_type

        def sig_cell(x_, m_, h_, w_i, w_h, b):
            h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
            h = T.nnet.sigmoid(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def tanh_cell(x_, m_, h_, w_i, w_h, b):
            h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
            h = T.tanh(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        def relu_cell(x_, m_, h_, w_i, w_h, b):
            h_pre = T.dot(h_, w_h) + T.dot(x_, w_i) + b
            h = T.nnet.relu(h_pre)
            h = m_[:, None] * h + (1. - m_[:, None]) * h_
            return h

        if self.nonlinearity == 'sigmoid':
            self.rnn_cell = sig_cell
        elif self.nonlinearity == 'tanh':
            self.rnn_cell = tanh_cell
        elif self.nonlinearity == 'relu':
            self.rnn_cell = relu_cell

    def inner(self, x, m, h_, tparams, nsteps):
        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'U')],
                                                tparams[_p(self.prefix, 'V')],
                                                tparams[_p(self.prefix, 'B')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates

class lstm_layer():
    def __init__(self, rnn_type, ninp, nhid):
        self.params = OrderedDict()
        #U_i = glorot_uniform([ninp, nhid])
        #U_f = glorot_uniform([ninp, nhid])
        #U_o = glorot_uniform([ninp, nhid])
        #U_g = glorot_uniform([ninp, nhid])

        #W_i = ortho_weight([nhid, nhid])
        #W_f = ortho_weight([nhid, nhid])
        #W_o = ortho_weight([nhid, nhid])
        #W_g = ortho_weight([nhid, nhid])

        bound = 0.02  # - 0.5
        U_i = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        U_f = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        U_o = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        U_g = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)

        W_i = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        W_f = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        W_o = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        W_g = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)

        self.params[_p(rnn_type, 'U_i')] = U_i
        self.params[_p(rnn_type, 'U_f')] = U_f
        self.params[_p(rnn_type, 'U_o')] = U_o
        self.params[_p(rnn_type, 'U_g')] = U_g

        self.params[_p(rnn_type, 'W_i')] = W_i
        self.params[_p(rnn_type, 'W_f')] = W_f
        self.params[_p(rnn_type, 'W_o')] = W_o
        self.params[_p(rnn_type, 'W_g')] = W_g

        self.prefix = rnn_type

        def cell(x_, m_, h_, c_, u_i, u_f, u_o, u_g, w_i, w_f, w_o, w_g):
            i = T.nnet.sigmoid(T.dot(x_, u_i) + T.dot(h_, w_i))
            f = T.nnet.sigmoid(T.dot(x_, u_f) + T.dot(h_, w_f))
            o = T.nnet.sigmoid(T.dot(x_, u_o) + T.dot(h_, w_o))
            g = T.tanh(T.dot(x_, u_g) + T.dot(h_, w_g))

            c = c_ * f + g * i
            h_pre = T.tanh(c) * o
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return [h, c]

        self.rnn_cell = cell

    def inner(self, x, m, h_, tparams, nsteps):
        [h, c], updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                      non_sequences=[tparams[_p(self.prefix, 'U_i')],
                                                     tparams[_p(self.prefix, 'U_f')],
                                                     tparams[_p(self.prefix, 'U_o')],
                                                     tparams[_p(self.prefix, 'U_g')],
                                                     tparams[_p(self.prefix, 'W_i')],
                                                     tparams[_p(self.prefix, 'W_f')],
                                                     tparams[_p(self.prefix, 'W_o')],
                                                     tparams[_p(self.prefix, 'W_g')]],
                                      outputs_info=[h_, h_], name=_p(self.prefix, '_layers'),
                                      n_steps=nsteps)
        return [h, c], updates



class gru_layer():
    def __init__(self, rnn_type, ninp, nhid):
        self.params = OrderedDict()
        #U_z = glorot_uniform([ninp, nhid])
        #U_r = glorot_uniform([ninp, nhid])
        #U_h = glorot_uniform([ninp, nhid])

        #W_z = ortho_weight([nhid, nhid])
        #W_r = ortho_weight([nhid, nhid])
        #W_h = ortho_weight([nhid, nhid])

        bound = 0.02  # - 0.5
        U_z = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        U_r = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)
        U_h = np.random.uniform(low=-bound, high=bound, size=([ninp, nhid])).astype(config.floatX)

        W_z = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        W_r = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)
        W_h = np.random.uniform(low=-bound, high=bound, size=([nhid, nhid])).astype(config.floatX)

        self.params[_p(rnn_type, 'U_z')] = U_z
        self.params[_p(rnn_type, 'U_r')] = U_r
        self.params[_p(rnn_type, 'U_h')] = U_h

        self.params[_p(rnn_type, 'W_z')] = W_z
        self.params[_p(rnn_type, 'W_r')] = W_r
        self.params[_p(rnn_type, 'W_h')] = W_h

        self.prefix = rnn_type

        def cell(x_, m_, h_, u_z, u_r, u_h, w_z, w_r, w_h):
            z = T.nnet.sigmoid(T.dot(x_, u_z) + T.dot(h_, w_z))
            r = T.nnet.sigmoid(T.dot(x_, u_r) + T.dot(h_, w_r))
            c = T.tanh(T.dot(x_, u_h) + T.dot(h_ * r, w_h))

            h_pre = (T.ones_like(z) - z) * c + z * h_
            h = m_[:, None] * h_pre + (1. - m_[:, None]) * h_
            return h

        self.rnn_cell = cell

    def inner(self, x, m, h_, tparams, nsteps):
        h, updates = theano.scan(self.rnn_cell, sequences=[x, m],
                                 non_sequences=[tparams[_p(self.prefix, 'U_z')],
                                                tparams[_p(self.prefix, 'U_r')],
                                                tparams[_p(self.prefix, 'U_h')],
                                                tparams[_p(self.prefix, 'W_z')],
                                                tparams[_p(self.prefix, 'W_r')],
                                                tparams[_p(self.prefix, 'W_h')]],
                                 outputs_info=h_, name=_p(self.prefix, '_layers'),
                                 n_steps=nsteps)
        return h, updates


class RNNModel():

    def __init__(self, rnn_type, ninp, nhid, nonlinearity, seed=666, lambda_value=0.5, debug=False):

        np.random.seed(seed)
        self.params = OrderedDict()

        if rnn_type == 'UNI':
            self.rnn = uni_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'O2':
            self.rnn = o2_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'M':
            self.rnn = m_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'MI':
            self.rnn = mi_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'SRN':
            self.rnn = srn_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'LSTM':
            self.rnn = lstm_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'GRU':
            self.rnn = gru_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        else:
            print('Model not available')
            exit(0)

        self.params = self.rnn.params

        self.prefix = rnn_type
        self.ninp = ninp
        self.nhid = nhid

        self.optimizer = rmsprop
        self.lambda_value = lambda_value
        self.debug = debug

        self.init_tparams()

    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.items():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)
            if self.debug:
                self.tparams[kk].tag.test_value = self.params[kk]
    def update_tparams(self, params=None):
        if params:
            for kk, vv in self.tparams.items():
                if kk not in params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(params[kk])
        else:
            for kk, vv in self.tparams.items():
                if kk not in self.params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(self.params[kk])

    def init_hidden(self, n_samples, seed):
        np.random.seed(seed)
        h_init_tmp = np.random.uniform(low=1e-5, high=1.0, size=(1, self.nhid-1)).astype(config.floatX)
        h_init = np.hstack((np.ones([1,1]), h_init_tmp)).astype(config.floatX)
        self.h_init = np.tile(h_init, (n_samples, 1))

    def reload_hidden(self, h_init, n_samples):
        self.h_init = np.tile(np.reshape(h_init,(1,-1)), (n_samples, 1))

    def build_model(self):

        lr = T.scalar(name='lr')
        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)
        y = T.vector('y', dtype='int32')

        if self.debug:
            lr.tag.test_value = np.random.rand(1)
            x.tag.test_value = np.random.randint(low=0, high=4, size=(50, self.h_init.shape[0], self.ninp), dtype='int32')
            x.tag.test_value = x.tag.test_value.astype(config.floatX)
            mask.tag.test_value = np.ones((50, self.h_init.shape[0]),
                                          dtype=config.floatX)
            mask.tag.test_value[:-5, :] = 0.0

            y.tag.test_value = np.random.randint(low=0, high=2,
                                                 size=(self.h_init.shape[0],),
                                                 dtype='int32')

        #n_timesteps = x.shape[0]
        #n_samples = x.shape[1]

        h = self.build_layer(x, mask)

        #loss = T.sum(((y - h[-1, :, 0]) ** 2) / 2)
        loss = T.sum((y - h[-1, :, 0]) ** 2)
        pred = h[-1, :, 0]
        grads = T.grad(grad_clip(loss, -1.0, 1.0), wrt=list(self.tparams.values()))
        #grads = T.grad(loss, wrt=list(self.tparams.values()))

        #self.f_states = theano.function([x, mask], outputs=h, name='f_states', profile=True)
        self.f_pred = theano.function([x, mask], outputs=pred, name='f_pred')#, profile=True)
        #self.f_pred.profile.summary()
        #self.f_grad = theano.function([x, mask, y], outputs=grads, name='f_grad', profile=True)
        self.f_grad_shared, self.f_update = \
            self.optimizer(lr, self.tparams, grads, x, mask, y, loss)

        self.f_pred.trust_input = True
        self.f_grad_shared.trust_input = True
        self.f_update.trust_input = True

    def build_layer(self, x, mask):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        if self.prefix != 'LSTM':
            h, updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return h
        else:
            [h, c], updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return T.concatenate((h, c), axis=-1)

class RNNModel_RNA():

    def __init__(self, rnn_type, ninp, nhid, nonlinearity, seed=666, lambda_value=0.5, debug=False):

        np.random.seed(seed)
        self.params = OrderedDict()

        if rnn_type == 'UNI':
            self.rnn = uni_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'O2':
            self.rnn = o2_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'M':
            self.rnn = m_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'MI':
            self.rnn = mi_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'SRN':
            self.rnn = srn_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'LSTM':
            self.rnn = lstm_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'GRU':
            self.rnn = gru_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        else:
            print('Model not available')
            exit(0)

        self.params = self.rnn.params

        W_out = glorot_uniform([nhid, 2])
        fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([nhid, 2]))
        bound = 1 / math.sqrt(fan_in)
        B_out = np.random.uniform(low=-bound, high=bound, size=2).astype(config.floatX)

        self.params[_p(rnn_type, 'W_out')] = W_out
        self.params[_p(rnn_type, 'B_out')] = B_out

        self.prefix = rnn_type
        self.ninp = ninp
        self.nhid = nhid

        self.optimizer = rmsprop
        self.lambda_value = lambda_value
        self.debug = debug

        self.init_tparams()

    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.items():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)
            if self.debug:
                self.tparams[kk].tag.test_value = self.params[kk]
    def update_tparams(self, params=None):
        if params:
            for kk, vv in self.tparams.items():
                if kk not in params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(params[kk])
        else:
            for kk, vv in self.tparams.items():
                if kk not in self.params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(self.params[kk])

    def init_hidden(self, n_samples, seed):
        np.random.seed(seed)
        h_init_tmp = np.random.uniform(low=1e-5, high=1.0, size=(1, self.nhid-1)).astype(config.floatX)
        h_init = np.hstack((np.ones([1,1]), h_init_tmp)).astype(config.floatX)
        self.h_init = np.tile(h_init, (n_samples, 1))

    def reload_hidden(self, h_init, n_samples):
        self.h_init = np.tile(np.reshape(h_init,(1,-1)), (n_samples, 1))

    def build_model(self):

        lr = T.scalar(name='lr')
        x = T.tensor3('x', dtype=config.floatX)
        mask = T.matrix('mask', dtype=config.floatX)
        y = T.vector('y', dtype='int32')

        if self.debug:
            lr.tag.test_value = np.random.rand(1)
            x.tag.test_value = np.random.randint(low=0, high=4, size=(16, self.h_init.shape[0], self.ninp), dtype='int32')
            x.tag.test_value = x.tag.test_value.astype(config.floatX)
            mask.tag.test_value = np.ones((16, self.h_init.shape[0]),
                                          dtype=config.floatX)
            mask.tag.test_value[:-5, :] = 0.0

            y.tag.test_value = np.random.randint(low=0, high=2,
                                                 size=(self.h_init.shape[0],),
                                                 dtype='int32')

        #n_timesteps = x.shape[0]
        #n_samples = x.shape[1]

        h = self.build_layer(x, mask)
        predout = T.dot(h[-1], self.tparams[_p(self.prefix, 'W_out')]) + self.tparams[_p(self.prefix, 'B_out')]
        out = T.nnet.softmax(predout)

        #loss = T.sum(((y - h[-1, :, 0]) ** 2) / 2)
        #loss = T.sum((y - h[-1, :, 0]) ** 2)
        loss = T.nnet.categorical_crossentropy(out, y).sum()

        pred = T.argmax(out,axis=1)
        grads = T.grad(grad_clip(loss, -1.0, 1.0), wrt=list(self.tparams.values()))
        #grads = T.grad(loss, wrt=list(self.tparams.values()))

        #self.f_states = theano.function([x, mask], outputs=h, name='f_states', profile=True)
        self.f_pred = theano.function([x, mask], outputs=pred, name='f_pred')#, profile=True)
        #self.f_pred.profile.summary()
        #self.f_grad = theano.function([x, mask, y], outputs=grads, name='f_grad', profile=True)
        self.f_grad_shared, self.f_update = \
            self.optimizer(lr, self.tparams, grads, x, mask, y, loss)

        self.f_pred.trust_input = True
        self.f_grad_shared.trust_input = True
        self.f_update.trust_input = True

    def build_layer(self, x, mask):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        if self.prefix != 'LSTM':
            h, updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return h
        else:
            [h, c], updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return T.concatenate((h, c), axis=-1)

class RNNModel_IMDB():

    def __init__(self, rnn_type, ntoken, ninp, nhid, nonlinearity, seed=666, lambda_value=2.0, debug=False):

        np.random.seed(seed)
        self.params = OrderedDict()

        if rnn_type == 'UNI':
            self.rnn = uni_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'O2':
            self.rnn = o2_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'M':
            self.rnn = m_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'MI':
            self.rnn = mi_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'SRN':
            self.rnn = srn_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid, nonlinearity=nonlinearity)
        elif rnn_type == 'LSTM':
            self.rnn = lstm_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        elif rnn_type == 'GRU':
            self.rnn = gru_layer(rnn_type=rnn_type, ninp=ninp, nhid=nhid)
        else:
            print('Model not available')
            exit(0)

        self.params = self.rnn.params

        W_emb = glorot_uniform([ntoken, ninp])
        self.params[_p(rnn_type, 'W_emb')] = W_emb

        W_out = glorot_uniform([nhid, 2])
        fan_in, _ = _calculate_fan_in_and_fan_out(np.zeros([nhid, 2]))
        bound = 1 / math.sqrt(fan_in)
        B_out = np.random.uniform(low=-bound, high=bound, size=2).astype(config.floatX)

        self.params[_p(rnn_type, 'W_out')] = W_out
        self.params[_p(rnn_type, 'B_out')] = B_out

        self.prefix = rnn_type
        self.ntoken = ntoken
        self.ninp = ninp
        self.nhid = nhid

        self.optimizer = adadelta#rmsprop
        self.lambda_value = lambda_value
        self.debug = debug

        self.init_tparams()

    def init_tparams(self):
        self.tparams = OrderedDict()
        for kk, pp in self.params.items():
            self.tparams[kk] = theano.shared(self.params[kk], name=kk)
            if self.debug:
                self.tparams[kk].tag.test_value = self.params[kk]
    def update_tparams(self, params=None):
        if params:
            for kk, vv in self.tparams.items():
                if kk not in params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(params[kk])
        else:
            for kk, vv in self.tparams.items():
                if kk not in self.params:
                    raise Warning('%s is not in the archive' % kk)
                self.tparams[kk].set_value(self.params[kk])

    def init_hidden(self, n_samples, seed):
        np.random.seed(seed)
        h_init_tmp = np.random.uniform(low=1e-5, high=1.0, size=(1, self.nhid-1)).astype(config.floatX)
        h_init = np.hstack((np.ones([1,1]), h_init_tmp)).astype(config.floatX)
        self.h_init = np.tile(h_init, (n_samples, 1))

    def reload_hidden(self, h_init, n_samples):
        self.h_init = np.tile(np.reshape(h_init,(1,-1)), (n_samples, 1))

    def build_model(self):

        lr = T.scalar(name='lr')
        x = T.matrix('x', dtype='int32')
        mask = T.matrix('mask', dtype=config.floatX)
        y = T.vector('y', dtype='int32')

        if self.debug:
            lr.tag.test_value = np.random.rand(1)
            x.tag.test_value = np.random.randint(low=0, high=4, size=(16, self.h_init.shape[0]), dtype='int32')
            mask.tag.test_value = np.ones((16, self.h_init.shape[0]), dtype=config.floatX)
            mask.tag.test_value[:-5, :] = 0.0
            y.tag.test_value = np.random.randint(low=0, high=2, size=(self.h_init.shape[0],), dtype='int32')



        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
        n_emb = self.tparams[_p(self.prefix, 'W_emb')].shape[1]

        #emb = self.tparams[_p(self.prefix, 'W_emb')][x.flatten()].reshape([n_timesteps, n_samples, n_emb])
        emb = self.tparams[_p(self.prefix, 'W_emb')][x.flatten()].reshape([n_timesteps, n_samples, self.ninp])

        h = self.build_layer(emb, mask)
        predout = T.dot(h[-1], self.tparams[_p(self.prefix, 'W_out')]) + self.tparams[_p(self.prefix, 'B_out')]
        out = T.nnet.softmax(predout)
        #out = T.nnet.sigmoid(predout)
        #loss = T.sum(((y - h[-1, :, 0]) ** 2) / 2)
        #loss = T.sum((y - h[-1, :, 0]) ** 2)
        loss = T.nnet.categorical_crossentropy(out, y).sum()

        if self.lambda_value > 0.:
            decay_c = theano.shared(numpy_floatX(self.lambda_value), name='decay_c')
            weight_decay = 0.
            if (self.prefix == 'O2') or (self.prefix == 'UNI'):
                weight_decay += (self.tparams[_p(self.prefix, 'W')] ** 2).sum()
            elif (self.prefix == 'MI') or (self.prefix == 'SRN'):
                weight_decay += (self.tparams[_p(self.prefix, 'V')] ** 2).sum()
            elif self.prefix == 'LSTM':
                weight_decay += (self.tparams[_p(self.prefix, 'W_i')] ** 2).sum() + \
                                (self.tparams[_p(self.prefix, 'W_f')] ** 2).sum() + \
                                (self.tparams[_p(self.prefix, 'W_o')] ** 2).sum() + \
                                (self.tparams[_p(self.prefix, 'W_g')] ** 2).sum()
            elif self.prefix == 'GRU':
                weight_decay += (self.tparams[_p(self.prefix, 'W_z')] ** 2).sum() + \
                                (self.tparams[_p(self.prefix, 'W_r')] ** 2).sum() + \
                                (self.tparams[_p(self.prefix, 'W_h')] ** 2).sum()

            weight_decay *= decay_c
            loss += weight_decay

        pred = T.argmax(out,axis=1)
        grads = T.grad(grad_clip(loss, -1.0, 1.0), wrt=list(self.tparams.values()))
        #grads = T.grad(loss, wrt=list(self.tparams.values()))

        #self.f_states = theano.function([x, mask], outputs=h, name='f_states', profile=True)
        self.f_pred = theano.function([x, mask], outputs=pred, name='f_pred')#, profile=True)
        #self.f_pred.profile.summary()
        #self.f_grad = theano.function([x, mask, y], outputs=grads, name='f_grad', profile=True)
        self.f_grad_shared, self.f_update = \
            self.optimizer(lr, self.tparams, grads, x, mask, y, loss)

        self.f_pred.trust_input = True
        self.f_grad_shared.trust_input = True
        self.f_update.trust_input = True

    def build_layer(self, x, mask):
        nsteps = x.shape[0]
        if x.ndim == 3:
            n_samples = x.shape[1]
        else:
            n_samples = 1

        if self.prefix != 'LSTM':
            h, updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return h
        else:
            [h, c], updates = self.rnn.inner(x, mask, self.h_init, self.tparams, nsteps)
            return h#T.concatenate((h, c), axis=-1)

'''
if __name__ == '__main__':

    model = RNNModel(rnn_type = 'o2_sig', ntoken=2, ninp=2, nhid=10,
                     ranSeed = 666, lambda_value = 0.5, debug = True)
    model.init_hidden(10)
    model.build_model()
'''
