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
# # Language Modeling & Sentiment Analysis of IMDB movie reviews

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.text import *

# %%
# bs=48
bs=128

# %%
path = untar_data(URLs.IMDB)

# %% [markdown]
# ## Language model

# %%
data_lm = (TextList.from_folder(path)
            .filter_by_folder(include=['train', 'test', 'unsup']) 
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()           
            .databunch(bs=bs, num_workers=1))

len(data_lm.vocab.itos),len(data_lm.train_ds)

# %%
data_lm.save('lm_databunch')

# %%
data_lm = load_data(path, 'lm_databunch', bs=bs)

# %%
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=1.).to_fp16()

# %%
lr = 1e-2
lr *= bs/48

# %%
learn_lm.fit_one_cycle(1, lr, moms=(0.8,0.7))

# %%
learn_lm.unfreeze()
learn_lm.fit_one_cycle(10, lr/10, moms=(0.8,0.7))

# %%
learn_lm.save('fine_tuned_10')
learn_lm.save_encoder('fine_tuned_enc_10')

# %% [markdown]
# ## Classifier

# %%
data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)
             .split_by_folder(valid='test')
             .label_from_folder(classes=['neg', 'pos'])
             .databunch(bs=bs, num_workers=1))

# %%
data_clas.save('imdb_textlist_class')

# %%
data_clas = load_data(path, 'imdb_textlist_class', bs=bs, num_workers=1)

# %%
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()
learn_c.load_encoder('fine_tuned_enc_10')
learn_c.freeze()

# %%
lr=2e-2
lr *= bs/48

# %%
learn_c.fit_one_cycle(1, lr, moms=(0.8,0.7))

# %%
learn_c.save('1')

# %%
learn_c.freeze_to(-2)
learn_c.fit_one_cycle(1, slice(lr/(2.6**4),lr), moms=(0.8,0.7))

# %%
learn_c.save('2nd')

# %%
learn_c.freeze_to(-3)
learn_c.fit_one_cycle(1, slice(lr/2/(2.6**4),lr/2), moms=(0.8,0.7))

# %%
learn_c.save('3rd')

# %%
learn_c.unfreeze()
learn_c.fit_one_cycle(2, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))

# %%
learn_c.save('clas')

# %%
