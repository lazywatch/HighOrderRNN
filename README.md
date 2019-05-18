# HighOrderRNN
Tomita, SL, SP, and STAMINA datasets are evaluated with RNNs implemented with Theano.
To run, make sure you have Theano=0.9.0 and numpy=1.11.3 and sklearn
```bash
conda install theano=0.9.0 numpy=1.11.3
```

## Tomita Grammars

Generate the data first.

```bash
python data_lstar.py --gram 1
```
The data_lstar.py is modified from the Training_Functions.py file in https://github.com/tech-srl/lstar_extraction

To train, run:
```bash
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 10 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn MI --nhid 8 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn M --nhid 6 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 7 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 5 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn GRU --nhid 5 --seed 1
python main_tomita.py --data g1 --epoch 100 --batch 100 --test_batch 10 --rnn LSTM --nhid 4 --seed 1
```

## SL and SP Grammars

Generate the data first.

```bash
python data_slsp.py --gram SL --k 4 --n 100k
```

To train, run:
```bash
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 30 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn MI --nhid 28 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn M --nhid 20 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 21 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn GRU --nhid 17 --seed 1
python main_slsp.py --data SL/SL4/100k/ --epoch 300 --batch 100 --test_batch 10 --rnn LSTM --nhid 14 --seed 1
```

## STAMINA Grammars

Generate the data first.

```bash
python data_stamina.py --number 81 --split 0.8
```

To train, run:
```bash
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn SRN --act tanh --nhid 100 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn MI --nhid 98 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn M --nhid 64 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn O2 --act tanh --nhid 17 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn UNI --act tanh --nhid 16 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn GRU --nhid 50 --seed 1
python main_stamina.py --data 81 --epoch 200 --batch 100 --test_batch 10 --rnn LSTM --nhid 41 --seed 1
```

## Penn Treebank - Word
Code are based on examples in <https://github.com/pytorch/examples/tree/master/word_language_model>.
All configurations and hyper paramters are centerized in a JSON file (`hps/penn.json` is an example for PTB).

find the conda(or miniconda)/env/ folder, create a virtual environment named py36torch, or other names you like.

```bash
conda create -n py36torch python=3.6
source activate py36torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
cd into the folder rnn_torch

For training, use

```bash
python main.py --hps-file hps/penn.json --model SRN-TANH
python main.py --hps-file hps/penn.json --model MI
python main.py --hps-file hps/penn.json --model M
python main.py --hps-file hps/penn.json --model O2
python main.py --hps-file hps/penn.json --model UNI
python main.py --hps-file hps/penn.json --model GRU
python main.py --hps-file hps/penn.json --model LSTM
```


