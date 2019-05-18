import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import json
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable

import data_ptb as data
from model_torch import RNNModel

###############################################################################
# Load configuration
###############################################################################
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN Language Model')
parser.add_argument('--hps-file', type=str, required=True,
                    help='location of hyper parameter json file.')
parser.add_argument('--model', type=str, default='UNI', help='rnn model')
args = parser.parse_args()

hps = json.load(open(args.hps_file))
hps['model'] = args.model

# Set the random seed manually for reproducibility.
torch.manual_seed(hps['seed'])
if torch.cuda.is_available():
    if not hps['cuda']:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        print("You run with --cuda")
        torch.cuda.manual_seed(hps['seed'])

###############################################################################
# Load data
###############################################################################
def batchify(data, bsz, cuda=False):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data

corpus = data.Corpus(hps['data'])
###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = RNNModel(hps['model'], ntokens, hps['emsize'], hps['nhid'],
                 hps['nlayers'], hps['dropout'], hps['tied'])
if hps['cuda']:
    model.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0) 
###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables,
        to detach them from their history."""
    if isinstance(h, Variable):
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(hps['bptt'], len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], requires_grad=False)#, volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1), requires_grad=False)
    return data, target


def evaluate(data_source):
    with torch.no_grad():
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0
        ntokens = len(corpus.dictionary)
        hidden = model.init_hidden(eval_batch_size)
        for i in range(0, data_source.size(0) - 1, hps['bptt']):
            data, targets = get_batch(data_source, i, evaluation=True)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(hps['batch_size'])
    for batch, i in enumerate(range(0, train_data.size(0) - 1, hps['bptt'])):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it
        # was previously produced. If we didn't, the model would try
        # backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem
        # in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), hps['clip'])
        #for p in model.parameters():
        #    p.data.add_(-lr, p.grad.data)
        optimizer.step()

        total_loss += loss.data

        if batch % hps['log_interval'] == 0 :# and batch > 0:
            cur_loss = total_loss.item() / hps['log_interval']
            elapsed = time.time() - start_time
            print(
                '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | '
                'ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // hps['bptt'], lr,
                    elapsed * 1000 / hps['log_interval'],
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



# Loop over epochs.
lr = hps['lr']
best_val_loss = None

eval_batch_size = 10
train_data = batchify(corpus.train, hps['batch_size'], hps['cuda'])
val_data = batchify(corpus.valid, eval_batch_size, hps['cuda'])
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, hps['epochs']+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(
                epoch, (time.time() - epoch_start_time),
                val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            os.makedirs(os.path.dirname(hps['save']), exist_ok=True)
            if hps['cuda']:
                torch.save(model.state_dict(), hps['save'] + '_' + hps['model'] + '.gpu')
                model.cpu()
                torch.save(model.state_dict(), hps['save'] + '_' + hps['model'] + '.cpu')
                model.cuda()
            else:
                torch.save(model.state_dict(), hps['save'] + '_' + hps['model'] + '.cpu')
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen
            # in the validation dataset.
            lr *=0.95
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

model = RNNModel(hps['model'], ntokens, hps['emsize'], hps['nhid'],
                 hps['nlayers'], hps['dropout'], hps['tied'])
# Load the best saved model params.
if hps['cuda']:
    model.cuda()
    model.load_state_dict(torch.load(hps['save'] + '_' + hps['model'] + '.gpu'))
else:
    model.load_state_dict(torch.load(hps['save'] + '_' + hps['model'] + '.cpu'))


# Run on test data.
test_data = batchify(corpus.test, eval_batch_size, hps['cuda'])
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)