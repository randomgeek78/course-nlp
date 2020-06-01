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
# # Vietnamese ULMFiT from scratch

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.text import *

# %%
# bs=48
# bs=24
bs=128

# %%
torch.cuda.set_device(2)

# %%
data_path = Config.data_path()

# %% [markdown]
# This will create a `viwiki` folder, containing a `viwiki` text file with the wikipedia contents. (For other languages, replace `vi` with the appropriate code from the [list of wikipedias](https://meta.wikimedia.org/wiki/List_of_Wikipedias).)

# %%
lang = 'vi'
# lang = 'zh'

# %%
name = f'{lang}wiki'
path = data_path/name
path.mkdir(exist_ok=True, parents=True)
lm_fns = [f'{lang}_wt', f'{lang}_wt_vocab']

# %% [markdown]
# ## Vietnamese wikipedia model

# %% [markdown] heading_collapsed=true
# ### Download data

# %% hidden=true
from nlputils import split_wiki,get_wiki

# %% hidden=true
get_wiki(path,lang)

# %% hidden=true
path.ls()

# %% hidden=true
# !head -n4 {path}/{name}

# %% [markdown] hidden=true
# This function splits the single wikipedia file into a separate file per article. This is often easier to work with.

# %% hidden=true
dest = split_wiki(path,lang)

# %% hidden=true
dest.ls()[:5]

# %% hidden=true
# Use this to convert Chinese traditional to simplified characters
# # ls *.txt | parallel -I% opencc -i % -o ../zhsdocs/% -c t2s.json

# %% [markdown] heading_collapsed=true
# ### Create pretrained model

# %% hidden=true
data = (TextList.from_folder(dest)
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()           
            .databunch(bs=bs, num_workers=1))

data.save(f'{lang}_databunch')
len(data.vocab.itos),len(data.train_ds)

# %% hidden=true
data = load_data(path, f'{lang}_databunch', bs=bs)

# %% hidden=true
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False).to_fp16()

# %% hidden=true
lr = 1e-2
lr *= bs/48  # Scale learning rate by batch size

# %% hidden=true
learn.unfreeze()
learn.fit_one_cycle(10, lr, moms=(0.8,0.7))

# %% [markdown] hidden=true
# Save the pretrained model and vocab:

# %% hidden=true
mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)
learn.to_fp32().save(mdl_path/lm_fns[0], with_opt=False)
learn.data.vocab.save(mdl_path/(lm_fns[1] + '.pkl'))

# %% [markdown]
# ## Vietnamese sentiment analysis

# %% [markdown]
# ### Language model

# %% [markdown]
# - [Data](https://github.com/ngxbac/aivivn_phanloaisacthaibinhluan/tree/master/data)
# - [Competition details](https://www.aivivn.com/contests/1)
# - Top 3 f1 scores: 0.900, 0.897, 0.897

# %%
train_df = pd.read_csv(path/'train.csv')
train_df.loc[pd.isna(train_df.comment),'comment']='NA'
train_df.head()

# %%
test_df = pd.read_csv(path/'test.csv')
test_df.loc[pd.isna(test_df.comment),'comment']='NA'
test_df.head()

# %%
df = pd.concat([train_df,test_df], sort=False)

# %%
data_lm = (TextList.from_df(df, path, cols='comment')
    .split_by_rand_pct(0.1, seed=42)
    .label_for_lm()           
    .databunch(bs=bs, num_workers=1))

# %%
learn_lm = language_model_learner(data_lm, AWD_LSTM, pretrained_fnames=lm_fns, drop_mult=1.0)

# %%
lr = 1e-3
lr *= bs/48

# %%
learn_lm.fit_one_cycle(2, lr*10, moms=(0.8,0.7))

# %%
learn_lm.unfreeze()
learn_lm.fit_one_cycle(8, lr, moms=(0.8,0.7))

# %%
learn_lm.save(f'{lang}fine_tuned')
learn_lm.save_encoder(f'{lang}fine_tuned_enc')

# %% [markdown]
# ### Classifier

# %%
data_clas = (TextList.from_df(train_df, path, vocab=data_lm.vocab, cols='comment')
    .split_by_rand_pct(0.1, seed=42)
    .label_from_df(cols='label')
    .databunch(bs=bs, num_workers=1))

data_clas.save(f'{lang}_textlist_class')

# %%
data_clas = load_data(path, f'{lang}_textlist_class', bs=bs, num_workers=1)

# %%
from sklearn.metrics import f1_score

@np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1))


# %%
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
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
learn_c.fit_one_cycle(1, slice(lr/10/(2.6**4),lr/10), moms=(0.8,0.7))

# %%
learn_c.save(f'{lang}clas')

# %% [markdown]
# Competition top 3 f1 scores: 0.90, 0.89, 0.89. Winner used an ensemble of 4 models: TextCNN, VDCNN, HARNN, and SARNN.

# %% [markdown]
# ## Ensemble

# %%
data_clas = load_data(path, f'{lang}_textlist_class', bs=bs, num_workers=1)
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c.load(f'{lang}clas', purge=False);

# %%
preds,targs = learn_c.get_preds(ordered=True)
accuracy(preds,targs),f1(preds,targs)

# %%
data_clas_bwd = load_data(path, f'{lang}_textlist_class_bwd', bs=bs, num_workers=1, backwards=True)
learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c_bwd.load(f'{lang}clas_bwd', purge=False);

# %%
preds_b,targs_b = learn_c_bwd.get_preds(ordered=True)
accuracy(preds_b,targs_b),f1(preds_b,targs_b)

# %%
preds_avg = (preds+preds_b)/2

# %%
accuracy(preds_avg,targs_b),f1(preds_avg,targs_b)

# %%
