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
# This notebook was part of [Lesson 7](https://course.fast.ai/videos/?lesson=7) of the Practical Deep Learning for Coders course.

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
# ### In deep learning, there are 2 types of numbers

# %% [markdown]
# **Parameters** are numbers that are learned.  **Activations** are numbers that are calculated (by affine functions & element-wise non-linearities).
#
# When you learn about any new concept in deep learning, ask yourself: is this a parameter or an activation?

# %% [markdown]
# Note to self: Point out the hidden state, going from the version without a for-loop to the for loop.  This is the step where people get confused.

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

# %%
len(data.valid_ds[0][0].data)

# %% [markdown]
# `bptt` stands for *back-propagation through time*.  This tells us how many steps of history we are considering.

# %%
data.bptt, len(data.valid_dl)

# %% [markdown]
# We have 3 batches in our validation set:
#
# 13017 tokens, with about ~70 tokens in about a line of text, and 64 lines of text per batch.

# %%
13017/70/bs

# %% [markdown]
# We will store each batch in a separate variable, so we can walk through this to understand better what the RNN does at each step:

# %%
it = iter(data.valid_dl)
x1,y1 = next(it)
x2,y2 = next(it)
x3,y3 = next(it)
it.close()

# %%
x1

# %% [markdown]
# `numel()` is a [PyTorch method](https://pytorch.org/docs/stable/torch.html#torch.numel) to return the number of elements in a tensor:

# %%
x1.numel()+x2.numel()+x3.numel()

# %%
x1.shape, y1.shape

# %%
x2.shape, y2.shape

# %%
x3.shape, y3.shape

# %%
v = data.valid_ds.vocab

# %%
v.itos

# %%
x1[:,0]

# %%
y1[:,0]

# %%
v.itos[9], v.itos[11], v.itos[12], v.itos[13], v.itos[10]

# %%
v.textify(x1[0])

# %%
v.textify(x1[1])

# %%
v.textify(x2[1])

# %%
v.textify(y1[0])

# %%
v.textify(x2[0])

# %%
v.textify(x3[0])

# %%
v.textify(x1[1])

# %%
v.textify(x2[1])

# %%
v.textify(x3[1])

# %%
v.textify(x3[-1])

# %%
data.show_batch(ds_type=DatasetType.Valid)

# %% [markdown]
# We will iteratively consider a few different models, building up to a more traditional RNN.

# %% [markdown]
# ## Single fully connected model

# %%
data = src.databunch(bs=bs, bptt=3)

# %%
x,y = data.one_batch()
x.shape,y.shape

# %%
nv = len(v.itos); nv

# %%
nh=64


# %%
def loss4(input,target): return F.cross_entropy(input, target[:,-1])
def acc4 (input,target): return accuracy(input, target[:,-1])


# %%
x[:,0]


# %% [markdown]
# Layer names:
# - `i_h`: input to hidden
# - `h_h`: hidden to hidden
# - `h_o`: hidden to output
# - `bn`: batchnorm

# %%
class Model0(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)  # green arrow
        self.h_h = nn.Linear(nh,nh)     # brown arrow
        self.h_o = nn.Linear(nh,nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)
        
    def forward(self, x):
        h = self.bn(F.relu(self.i_h(x[:,0])))
        if x.shape[1]>1:
            h = h + self.i_h(x[:,1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1]>2:
            h = h + self.i_h(x[:,2])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


# %%
learn = Learner(data, Model0(), loss_func=loss4, metrics=acc4)

# %%
learn.fit_one_cycle(6, 1e-4)


# %% [markdown]
# ## Same thing with a loop

# %% [markdown]
# Let's refactor this to use a for-loop.  This does the same thing as before:

# %%
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)  # green arrow
        self.h_h = nn.Linear(nh,nh)     # brown arrow
        self.h_o = nn.Linear(nh,nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)
        
    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


# %% [markdown]
# This is the difference between unrolled (what we had before) and rolled (what we have now) RNN diagrams:

# %%
learn = Learner(data, Model1(), loss_func=loss4, metrics=acc4)

# %%
learn.fit_one_cycle(6, 1e-4)

# %% [markdown]
# Our accuracy is about the same, since we are doing the same thing as before.

# %% [markdown]
# ## Multi fully connected model

# %% [markdown]
# Before, we were just predicting the last word in a line of text.  Given 70 tokens, what is token 71?  That approach was throwing away a lot of data.  Why not predict token 2 from token 1, then predict token 3, then predict token 4, and so on?  We will modify our model to do this.

# %%
data = src.databunch(bs=bs, bptt=20)

# %%
x,y = data.one_batch()
x.shape,y.shape


# %%
class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        
    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)


# %%
learn = Learner(data, Model2(), metrics=accuracy)

# %%
learn.fit_one_cycle(10, 1e-4, pct_start=0.1)


# %% [markdown]
# Note that our accuracy is worse now, because we are doing a harder task.  When we predict word k (k<70), we have less history to help us then when we were only predicting word 71.

# %% [markdown]
# ## Maintain state

# %% [markdown]
# To address this issue, let's keep the hidden state from the previous line of text, so we are not starting over again on each new line of text.

# %%
class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.h_h = nn.Linear(nh,nh)
        self.h_o = nn.Linear(nh,nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()
        
    def forward(self, x):
        res = []
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res


# %%
learn = Learner(data, Model3(), metrics=accuracy)

# %%
learn.fit_one_cycle(20, 3e-3)


# %% [markdown]
# Now we are getting greater accuracy than before!

# %% [markdown]
# ## nn.RNN

# %% [markdown]
# Let's refactor the above to use PyTorch's RNN.  This is what you would use in practice, but now you know the inside details!

# %%
class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.RNN(nh,nh, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(1, bs, nh).cuda()
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))


# %%
learn = Learner(data, Model4(), metrics=accuracy)

# %%
learn.fit_one_cycle(20, 3e-3)


# %% [markdown]
# ## 2-layer GRU

# %% [markdown]
# When you have long time scales and deeper networks, these become impossible to train.  One way to address this is to add mini-NN to decide how much of the green arrow and how much of the orange arrow to keep.  These mini-NNs can be GRUs or LSTMs.  We will cover more details of this in a later lesson.

# %%
class Model5(nn.Module):
    def __init__(self):
        super().__init__()
        self.i_h = nn.Embedding(nv,nh)
        self.rnn = nn.GRU(nh, nh, 2, batch_first=True)
        self.h_o = nn.Linear(nh,nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(2, bs, nh).cuda()
        
    def forward(self, x):
        res,h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))


# %%
learn = Learner(data, Model5(), metrics=accuracy)

# %%
learn.fit_one_cycle(10, 1e-2)

# %% [markdown]
# ### Connection to ULMFit

# %% [markdown]
# In the previous lesson, we were essentially swapping out `self.h_o` with a classifier in order to do classification on text.

# %% [markdown]
# ## fin

# %% [markdown]
# RNNs are just a refactored, fully-connected neural network.
#
# You can use the same approach for any sequence labeling task (part of speech, classifying whether material is sensitive,..)
