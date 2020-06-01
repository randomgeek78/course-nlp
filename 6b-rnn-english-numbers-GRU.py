# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting English word version of numbers using an RNN

# %% [markdown]
# We were using RNNs as part of our language model in the previous lesson.  Today, we will dive into more details of what RNNs are and how they work.  We will do this using the problem of trying to predict the English word version of numbers.
#
# Let's predict what should come next in this sequence:
#
# *eight thousand one , eight thousand two , eight thousand three , eight thousand four , eight thousand five , eight thousand six , eight thousand seven , eight thousand eight , eight thousand nine , eight thousand ten , eight thousand eleven , eight thousand twelve...*
#
#
# Jeremy created this synthetic dataset to have a better way to check if things are working, to debug, and to understand what was going on. When experimenting with new ideas, it can be nice to have a smaller dataset to do so, to quickly get a sense of whether your ideas are promising (for other examples, see [Imagenette and Imagewoof](https://github.com/fastai/imagenette)) This English word numbers will serve as a good dataset for learning about RNNs.  Our task today will be to predict which word comes next when counting.

# %% [markdown]
# ## Data

# %%
from fastai.text import *

# %%
bs=64

# %%
path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()


# %%
def readnums(d): return [', '.join(o.strip() for o in open(path/d).readlines())]


# %% [markdown]
# train.txt gives us a sequence of numbers written out as English words:

# %%
train_txt = readnums('train.txt'); train_txt[0][:80]

# %%
valid_txt = readnums('valid.txt'); valid_txt[0][-80:]

# %%
train = TextList(train_txt, path=path)
valid = TextList(valid_txt, path=path)

src = ItemLists(path=path, train=train, valid=valid).label_for_lm()
data = src.databunch(bs=bs)

# %%
train[0].text[:80]

# %% [markdown]
# `bptt` stands for *back-propagation through time*.  This tells us how many steps of history we are considering.

# %%
data.bptt, len(data.valid_dl)

# %% [markdown]
# We have 3 batches in our validation set:
#
# 13017 tokens, with about ~70 tokens in about a line of text, and 64 lines of text per batch.

# %% [markdown]
# We will store each batch in a separate variable, so we can walk through this to understand better what the RNN does at each step:

# %%
it = iter(data.valid_dl)
x1,y1 = next(it)
x2,y2 = next(it)
x3,y3 = next(it)
it.close()

# %%
v = data.valid_ds.vocab

# %%
data = src.databunch(bs=bs, bptt=40)

# %%
x,y = data.one_batch()
x.shape,y.shape

# %%
nv = len(v.itos); nv

# %%
nh=56


# %%
def loss4(input,target): return F.cross_entropy(input, target[:,-1])
def acc4 (input,target): return accuracy(input, target[:,-1])


# %% [markdown]
# Layer names:
# - `i_h`: input to hidden
# - `h_h`: hidden to hidden
# - `h_o`: hidden to output
# - `bn`: batchnorm

# %% [markdown]
# ## Adding a GRU

# %% [markdown]
# When you have long time scales and deeper networks, these become impossible to train.  One way to address this is to add mini-NN to decide how much of the green arrow and how much of the orange arrow to keep.  These mini-NNs can be GRUs or LSTMs.

# %%
class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.GRU(nh, nh, 1, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(1, bs, nh).cuda()
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))


# %%
nv, nh

# %%
learn = Learner(data, Model5(), metrics=accuracy)

# %%
learn.fit_one_cycle(10, 1e-2)


# %% [markdown]
# ## Let's make our own GRU

# %% [markdown]
# ### Using PyTorch's GRUCell

# %% [markdown]
# Axis 0 is the batch dimension, and axis 1 is the time dimension.  We want to loop through axis 1:

# %%
def rnn_loop(cell, h, x):
    res = []
    for x_ in x.transpose(0,1):
        h = cell(x_, h)
        res.append(h)
    return torch.stack(res, dim=1)


# %%
class Model6(Model5):
    def __init__(self):
        super().__init__()
        self.rnnc = nn.GRUCell(nh, nh)
        self.h = torch.zeros(bs, nh).cuda()
        
    def forward(self, x):
        res = rnn_loop(self.rnnc, self.h, self.i_h(x))
        self.h = res[:,-1].detach()
        return self.h_o(self.bn(res))


# %%
learn = Learner(data, Model6(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-2)


# %% [markdown]
# ### With a custom GRUCell

# %% [markdown]
# The following is based on code from [emadRad](https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb):

# %%
class GRUCell(nn.Module):
    def __init__(self, ni, nh):
        super(GRUCell, self).__init__()
        self.ni,self.nh = ni,nh
        self.i2h = nn.Linear(ni, 3*nh)
        self.h2h = nn.Linear(nh, 3*nh)
    
    def forward(self, x, h):
        gate_x = self.i2h(x).squeeze()
        gate_h = self.h2h(h).squeeze()
        i_r,i_u,i_n = gate_x.chunk(3, 1)
        h_r,h_u,h_n = gate_h.chunk(3, 1)
        
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_u + h_u)
        newgate = torch.tanh(i_n + (resetgate*h_n))
        return updategate*h + (1-updategate)*newgate


# %%
class Model7(Model6):
    def __init__(self):
        super().__init__()
        self.rnnc = GRUCell(nh,nh)


# %%
learn = Learner(data, Model7(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-2)

# %% [markdown]
# ### Connection to ULMFit

# %% [markdown]
# In the previous lesson, we were essentially swapping out `self.h_o` with a classifier in order to do classification on text.
#
# RNNs are just a refactored, fully-connected neural network.
#
# You can use the same approach for any sequence labeling task (part of speech, classifying whether material is sensitive,..)

# %% [markdown]
# ## fin
