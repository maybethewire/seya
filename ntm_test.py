from __future__ import absolute_import

import logging
import numpy as np
np.random.seed(124)
import matplotlib.pyplot as plt
import cPickle

from theano import tensor, function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras.utils import np_utils, generic_utils
from keras.optimizers import Adam, SGD
from keras import backend as K

from seya.layers.ntm import NeuralTuringMachine as NTM
#from seya.models import Sequential  # this is just good old Sequential, from before TensorFlow support


batch_size = 100

h_dim = 64
n_slots = 50
m_length = 20
input_dim = 8
lr = 1e-3
clipnorm = 10

# Neural Turing Machine

ntm = NTM(h_dim, n_slots=n_slots, m_length=m_length, shift_range=3,
          inner_rnn='gru', return_sequences=True, input_dim=input_dim, input_length=None)
model = Sequential()
#model.add(Masking(input_shape=(None, input_dim)))
model.add(ntm)
model.add(TimeDistributed(Dense(input_dim)))
model.add(Activation('sigmoid'))

sgd = Adam(lr=lr, clipnorm=clipnorm)
model.compile(loss='binary_crossentropy', optimizer=sgd, sample_weight_mode="temporal")

model.summary()

# %%
def get_sample(batch_size=128, n_bits=8, max_size=20, min_size=1):
    # generate samples with random length
    inp = np.zeros((batch_size, 2*max_size-1, n_bits))
    out = np.zeros((batch_size, 2*max_size-1, n_bits))
    sw = np.zeros((batch_size, 2*max_size-1, 1))
    for i in range(batch_size):
        t = np.random.randint(low=min_size, high=max_size)
        x = np.random.uniform(size=(t, n_bits)) > .5
        for j,f in enumerate(x.sum(axis=-1)): # remove fake flags
            if f>=n_bits:
                x[j, :] = 0.
        del_flag = np.ones((1, n_bits))
        inp[i, :t+1] = np.concatenate([x, del_flag], axis=0)
        out[i, t:(2*t)] = x
        sw[i, t:(2*t)] = 1
    return inp, out, sw
    

def save_pattern(inp, out, sw, file_name='pattern2.png'):
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(inp>.5)
    plt.subplot(132)
    plt.imshow(out>.5)
    plt.subplot(133)
    plt.imshow(sw[:, :1]>.5)
    plt.savefig(file_name)
    plt.close()

def show_pattern(inp, out, sw):
    plt.subplot(131)
    plt.title('input/output')
    plt.imshow(inp, cmap='gray')
    plt.subplot(132)
    plt.title('desired')
    plt.imshow(out, cmap='gray')
    plt.subplot(133)
    plt.title('sample_weight')
    plt.imshow(sw, cmap='gray')
  
inp, out, sw = get_sample(1, 8, 20, 19)
#show_pattern(inp[0], out[0], sw[0])

    
# %%
# sample_weight marks the points in time that will 
# be part of the cost function.

# training uses sequences of length 1 to 20. Test uses series of the same length.
def test_model(model, min_size=19):
    I, V, sw = get_sample(batch_size=500, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    Y = np.asarray(model.predict(I, batch_size=100) > .5).astype('float64')
    acc = (V[:, -min_size:, :] == Y[:, -min_size:, :]).mean() * 100
    #show_pattern(Y[0], V[0], sw[0])

    return acc
    
trained = model

nb_epoch = 4000
progbar = generic_utils.Progbar(nb_epoch)
ACC = []
ac = np.nan
for e in range(nb_epoch):
    I, V, sw = get_sample(n_bits=input_dim, max_size=20, min_size=1, batch_size=128)
    #loss = trained.train_on_batch(I, V, sample_weight=sw[:, :, 0])[0]
    hist = trained.fit(I, V, sample_weight=sw[:, :, 0], nb_epoch=1, batch_size=128, verbose=0)
    loss = hist.history['loss'][-1]
    progbar.add(1, values=[('loss', loss)])
    
    if e % 500 == 0:
        print("")
        acc = test_model(trained)
        l = []
        for a in [acc,]:
            print("acc: {}".format(a))
            l.append(a)
        ACC.append(l)
        
# %%
inp, out, sw = get_sample(1, 8, 20, 19)
#plt.subplot(131)
#plt.title('input')
#plt.imshow(inp[0], cmap='gray')
#plt.subplot(132)
#plt.title('desired')
#plt.imshow(out[0], cmap='gray')
#plt.subplot(133)
#plt.title('sample_weight')
#plt.imshow(sw[0], cmap='gray')

# %%
p = model.predict(inp)

#Y = ntm.get_full_output()[0:3] # (memory over time, read_vectors, write_vectors)
#F = function([X], Y, allow_input_downcast=True)
#
#inp, out, sw = get_sample(1, input_dim, 31, 30)
#
#mem = mem.transpose(1, 0, 2).reshape((1, -1, n_slots, m_length))
#write = write.transpose(1, 0, 2)
#read = read.transpose(1, 0, 2)


        
