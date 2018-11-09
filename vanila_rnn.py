"""
Minimal character-level Vanilla RNN model. Written b_y Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import unicodedata
import string
import codecs

# data I/O
data = codecs.open('data/potter.txt', 'r', encoding='utf8', errors='ignore').read()
fake = codecs.open('data/output.txt', 'w', encoding='utf8')
chars = list(set(data))
data_size = len(data)                                                       # 
vocab_size = len(chars)

print(f'data has {data_size} characters,{vocab_size} unique.')              # data has 1109177 characters,80 unique.

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print(char_to_ix)
print(ix_to_char)

# hyperparameters
hidden_size = 256                                                           # size of hidden layer of neurons
seq_length = 128                                                            # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
W_xh = np.random.randn(hidden_size, vocab_size) * 0.01                      # weight: input to hidden
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01                     # weight: hidden to hidden
W_hy = np.random.randn(vocab_size, hidden_size) * 0.01                      # weight: hidden to output
b_h = np.zeros((hidden_size, 1))                                            # hidden bias
b_y = np.zeros((vocab_size, 1))                                             # output bias


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def lossFun(inputs, targets, hprev):
    """
    inputs,targets are both list of integers indicating which unique character.
        inputs: a seq_length size list
    hprev is (H x 1) array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}                                         # sx[t] = ys[t] = ps[t] size = vocab_size x 1
    hs[-1] = np.copy(hprev)                                                 # hs[t] size = hidden_size * 1
    loss = 0                                                                # xs: input line; ys: output line; hs: hidden states, multiple of them,
                                                                            # even the weights are reused, the states are different from each other.

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1))                                    # encode in 1-of-k representation
        xs[t][inputs[t]] = 1                                                # inputs[t] is a index number, xs[t] is a vector
        hs[t] = np.tanh(np.dot(W_xh, xs[t]) + np.dot(W_hh, hs[t-1]) + b_h)  # hidden state
        ys[t] = np.dot(W_hy, hs[t]) + b_y                                   # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))                       # (normalized) probabilities for next chars
        loss += -np.log(ps[t][targets[t],0])                                # softmax (cross-entropy loss)
        print(f'loss: {loss}')
        # print(f'xs:{len(xs[t])}->{xs[t]}\n hs:{len(hs[t])}->{hs[t]}\n ys:{len(ys[t])}->{ys[t]}\n ps:{len(ps[t])}->{ps[t]}')

    # backward pass: compute gradients going backwards
    dW_xh = np.zeros_like(W_xh)                                             # gradient of W_xh, same shape as W_xh
    dW_hh = np.zeros_like(W_hh)                                             # gradient of W_hh, same shape as W_hh
    dW_hy = np.zeros_like(W_hy)                                             # gradient of W_hy, same shape as W_hy
    db_h = np.zeros_like(b_h)                                               # gradient of b_h, same shape as b_h
    db_y = np.zeros_like(b_y)                                               # gradient of b_y, same shape as b_y
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
        dW_hy += np.dot(dy, hs[t].T)
        db_y += dy
        dh = np.dot(W_hy.T, dy) + dhnext                                    # backprop into h
        dhraw = (1 - hs[t] * hs[t]) * dh                                    # backprop through tanh nonlinearity
        db_h += dhraw
        dW_xh += np.dot(dhraw, xs[t].T)
        dW_hh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(W_hh.T, dhraw)

    for dparam in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
        np.clip(dparam, -5, 5, out=dparam)                                  # clip to mitigate exploding gradients
    return loss, dW_xh, dW_hh, dW_hy, db_h, db_y, hs[len(inputs)-1]


def sample(h, seed_ix, n):
    """ 
    sample a sequence of integers from the model 
    h is memory state, seed_ix is seed letter for first time step
    i.e. do predictions :)
    """
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
        h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h)
        y = np.dot(W_hy, h) + b_y
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes



n, p = 0, 0
mW_xh, mW_hh, mW_hy = np.zeros_like(W_xh), np.zeros_like(W_hh), np.zeros_like(W_hy)
mb_h, mb_y = np.zeros_like(b_h), np.zeros_like(b_y)                         # memory variables for Adagrad
smooth_loss = -np.log(1.0 / vocab_size) * seq_length                        # loss at iteration 0
while True:
    try:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= len(data) or n == 0: 
            hprev = np.zeros((hidden_size,1))                               # reset RNN memory
            p = 0                                                           # go from start of data
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample(hprev, inputs[0], 200)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt, ))

        # forward seq_length characters through the net and fetch gradient
        loss, dW_xh, dW_hh, dW_hy, db_h, db_y, hprev = lossFun(inputs, targets, hprev)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        if n % 100 == 0:
          print(f'iter{n}, loss: {smooth_loss}')                            # print progress
        
        # perform parameter update with Adagrad
        for param, dparam, mem in zip([W_xh, W_hh, W_hy, b_h, b_y], 
                                      [dW_xh, dW_hh, dW_hy, db_h, db_y], 
                                      [mW_xh, mW_hh, mW_hy, mb_h, mb_y]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)          # adagrad update

        p += seq_length                                                     # move data pointer
        n += 1                                                              # iteration counter
    except KeyboardInterrupt:
        sample_ix = sample(hprev, inputs[0], data_size)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        fake.write(txt)
        break
fake.close()