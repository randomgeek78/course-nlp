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
# # Turkish ULMFiT from scratch

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.text import *

# %%
bs=128
torch.cuda.set_device(2)
data_path = Config.data_path()

lang = 'tr'
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)

# %%
mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)
lm_fns = [mdl_path/f'{lang}_wt', mdl_path/f'{lang}_wt_vocab']

# %% [markdown]
# ## Turkish wikipedia model

# %%
from nlputils import split_wiki,get_wiki

get_wiki(path,lang)
# !head -n4 {path}/{name}

# %%
dest = split_wiki(path,lang)

# %% [markdown]
# Turkish is an [Agglutinative_language](https://en.wikipedia.org/wiki/Agglutinative_language) so it needs special care!
#
# ![Turkish morphemes example](images/turkish.jpg)

# %%
data = (TextList.from_folder(dest, processor=[OpenFileProcessor(), SPProcessor()])
        .split_by_rand_pct(0.1, seed=42)
        .label_for_lm()
        .databunch(bs=bs, num_workers=1))

data.save(f'{lang}_databunch')
len(data.vocab.itos),len(data.train_ds)

# %%
data = load_data(dest, f'{lang}_databunch', bs=bs)

# %%
data.show_batch()

# %%
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.1, wd=0.1, pretrained=False).to_fp16()

# %%
lr = 3e-3
lr *= bs/48  # Scale learning rate by batch size

# %%
learn.unfreeze()
learn.fit_one_cycle(10, lr, moms=(0.8,0.7))

# %%
learn.to_fp32().save(lm_fns[0], with_opt=False)
learn.data.vocab.save(lm_fns[1].with_suffix('.pkl'))

# %% [markdown]
# ## Turkish sentiment analysis

# %% [markdown]
# https://www.win.tue.nl/~mpechen/projects/smm/

# %% [markdown]
# ### Language model

# %%
path_clas = path/'movies'
path_clas.ls()

# %%
pos = (path_clas/'tr_polarity.pos').open(encoding='iso-8859-9').readlines()
pos_df = pd.DataFrame({'text':pos})
pos_df['pos'] = 1
pos_df.head()

# %%
neg = (path_clas/'tr_polarity.neg').open(encoding='iso-8859-9').readlines()
neg_df = pd.DataFrame({'text':neg})
neg_df['pos'] = 0
neg_df.head()

# %%
df = pd.concat([pos_df,neg_df], sort=False)

# %%
data_lm = (TextList.from_df(df, path_clas, cols='text', processor=SPProcessor.load(dest))
    .split_by_rand_pct(0.1, seed=42)
    .label_for_lm()           
    .databunch(bs=bs, num_workers=1))

data_lm.save(f'{lang}_clas_databunch')

# %%
data_lm = load_data(path_clas, f'{lang}_clas_databunch', bs=bs)

# %%
data_lm.show_batch()

# %%
learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=1.0, wd=0.1)

# %%
lr = 1e-3
lr *= bs/48

# %%
learn_lm.fit_one_cycle(1, lr*10, moms=(0.8,0.7))

# %%
learn_lm.unfreeze()
learn_lm.fit_one_cycle(5, slice(lr/10,lr*10), moms=(0.8,0.7))

# %%
learn_lm.save(f'{lang}fine_tuned')
learn_lm.save_encoder(f'{lang}fine_tuned_enc')

# %% [markdown]
# ### Classifier

# %%
data_clas = (TextList.from_df(df, path_clas, cols='text', processor=SPProcessor.load(dest))
    .split_by_rand_pct(0.1, seed=42)
    .label_from_df(cols='pos')
    .databunch(bs=bs, num_workers=1))

# %%
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, pretrained=False, wd=0.1).to_fp16()
learn_c.load_encoder(f'{lang}fine_tuned_enc')
learn_c.freeze()

# %%
lr=2e-2
lr *= bs/48

# %%
learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))

# %%
learn_c.fit_one_cycle(2, lr, moms=(0.8,0.7))

# %%
learn_c.freeze_to(-2)
learn_c.fit_one_cycle(2, slice(lr/(2.6**4),lr), moms=(0.8,0.7))

# %%
learn_c.freeze_to(-3)
learn_c.fit_one_cycle(2, slice(lr/2/(2.6**4),lr/2), moms=(0.8,0.7))

# %%
learn_c.unfreeze()
learn_c.fit_one_cycle(4, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))

# %% [markdown]
# Accuracy in Gezici (2018), *Sentiment Analysis in Turkish* is: `75.16%`.

# %%
learn_c.save(f'{lang}clas')

# %% [markdown]
# ## fin

# %%
