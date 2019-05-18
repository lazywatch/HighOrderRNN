import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class UNIRNNlayer(nn.Module):
    r"""Applies a tensor transformation to the incoming data: :math:`h = xWh + Ux + Vh + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(UNIRNNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(in_features, out_features, out_features)
        for i in range(in_features):
            for j in range(out_features):
                init.orthogonal_(weight[i, j:j + 1])
        self.W = Parameter(weight.view(in_features, -1))

        #self.W = Parameter(torch.empty(in_features, out_features * out_features))
        self.U = Parameter(torch.empty(in_features, out_features))
        self.V = Parameter(torch.empty(out_features, out_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.U, a=math.sqrt(5))
        init.orthogonal_(self.V)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.U.t())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, hidden=None, reverse=False):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = self.inner(input[i], hidden)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    @weak_script_method
    def inner(self, input, hidden=None):

        WX = F.linear(input, self.W.transpose(1, 0))  # Compute WX = W dot X.
        WX = WX.view(-1, self.out_features, self.out_features)  # Compute WX = W dot X.
        WHX = WX.bmm(hidden.unsqueeze(2)).squeeze(2)  # Compute WHX = WX dot H.
        UX = F.linear(input, self.U.t())  # Compute UX = U dot X.
        VH = F.linear(hidden, self.V.t())  # Compute VH = V dot H.

        if self.bias is None:
            return torch.tanh(WHX + UX + VH)
        else:
            return torch.tanh(WHX + UX + VH + self.bias)


    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


@weak_module
class O2RNNlayer(nn.Module):
    r"""Applies a tensor transformation to the incoming data: :math:`h = xWh + Ux + Vh + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=False):
        super(O2RNNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(in_features, out_features, out_features)
        for i in range(in_features):
            for j in range(out_features):
                init.orthogonal_(weight[i, j:j+1])
        self.weight = Parameter(weight.view(in_features, -1))

        #self.weight = Parameter(torch.empty(in_features, out_features * out_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        #init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        #init.uniform_(self.weight, a=0.99, b=1.01)
        #init.orthogonal_(self.weight)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(torch.empty(self.in_features, self.out_features))
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            #self.bias.data.fill_(2)

    def forward(self, input, hidden=None, reverse=False):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = self.inner(input[i], hidden)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    @weak_script_method
    def inner(self, input, hidden=None):

        WX = F.linear(input, self.weight.transpose(1, 0))  # Compute WX = W dot X.
        WX = WX.view(-1, self.out_features, self.out_features)  # Compute WX = W dot X.
        WHX = WX.bmm(hidden.unsqueeze(2)).squeeze(2)  # Compute WHX = WX dot H.
        #aa = torch.sigmoid(WHX + self.bias)
        if self.bias is not None:
            return torch.tanh(WHX + self.bias)
        else:
            return torch.tanh(WHX)
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

@weak_module
class MRNNlayer(nn.Module):
    r"""Applies a tensor transformation to the incoming data: :math:`h = xWh + Ux + Vh + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(MRNNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fx = Parameter(torch.empty(in_features, out_features))
        self.fh = Parameter(torch.empty(out_features, out_features))
        self.hf = Parameter(torch.empty(out_features, out_features))
        self.hx = Parameter(torch.empty(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.fx, a=math.sqrt(5))
        init.orthogonal_(self.fh)
        init.orthogonal_(self.hf)
        init.kaiming_uniform_(self.hx, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.fx.t())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input, hidden=None, reverse=False):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = self.inner(input[i], hidden)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    @weak_script_method
    def inner(self, input, hidden=None):

        tmp0 = torch.diag_embed(F.linear(input, self.fx.t()))  # Compute UX = U dot X.
        tmp1 = F.linear(hidden, self.fh.t())  # Compute VH = V dot H.
        ft = tmp0.bmm(tmp1.unsqueeze(2)).squeeze(2)

        return F.tanh(F.linear(ft, self.hf.t()) + F.linear(input, self.hx.t()) + self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


@weak_module
class MIRNNlayer(nn.Module):
    r"""Applies a tensor transformation to the incoming data: :math:`h = xWh + Ux + Vh + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, \text{in\_features})` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, \text{out\_features})` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, bias=True):
        super(MIRNNlayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.U = Parameter(torch.empty(in_features, out_features))
        self.V = Parameter(torch.empty(out_features, out_features))

        if bias:
            self.bias = Parameter(torch.empty(out_features))
            self.alpha = Parameter(torch.empty(out_features))
            self.beta1 = Parameter(torch.empty(out_features))
            self.beta2 = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.U, a=math.sqrt(5))
        init.orthogonal_(self.V)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.U.t())
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            init.uniform_(self.alpha, -bound, bound)
            init.uniform_(self.beta1, -bound, bound)
            init.uniform_(self.beta2, -bound, bound)

    def forward(self, input, hidden=None, reverse=False):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = self.inner(input[i], hidden)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    @weak_script_method
    def inner(self, input, hidden=None):

        UX = F.linear(input, self.U.t())  # Compute UX = U dot X.
        VH = F.linear(hidden, self.V.t())  # Compute VH = V dot H.

        return torch.tanh(self.alpha * UX * VH + self.beta1 * VH + self.beta2 * UX + self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class UNIRNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5, rnn_type='UNI', **kwargs):
        super(UNIRNN, self).__init__()

        if rnn_type == 'UNI':
            self.layers = nn.ModuleList(
                [UNIRNNlayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
        elif rnn_type == 'O2':
            self.layers = nn.ModuleList(
                [O2RNNlayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
        elif rnn_type == 'M':
            self.layers = nn.ModuleList(
                [MRNNlayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])
        elif rnn_type == 'MI':
            self.layers = nn.ModuleList(
                [MIRNNlayer(input_size if l == 0 else hidden_size, hidden_size, **kwargs) for l in range(num_layers)])

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, input, hidden=None):
        next_hidden = []

        for i, layer in enumerate(self.layers):
            hn, input = layer(input, None if hidden is None else hidden[i])
            next_hidden.append(hn)

            if self.dropout != 0 and i < len(self.layers) - 1:
                input = F.dropout(input, p=self.dropout, training=self.training, inplace=False)

        next_hidden = torch.cat(next_hidden, 0).view(self.num_layers, *next_hidden[0].size())

        return input, next_hidden

