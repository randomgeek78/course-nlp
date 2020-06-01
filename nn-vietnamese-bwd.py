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
# # Vietnamese ULMFiT from scratch (backwards)

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.text import *

# %%
bs=128

# %%
data_path = Config.data_path()
lang = 'vi'
name = f'{lang}wiki'
path = data_path/name
dest = path/'docs'
lm_fns = [f'{lang}_wt_bwd', f'{lang}_wt_vocab_bwd']

# %% [markdown]
# ## Vietnamese wikipedia model

# %%
data = (TextList.from_folder(dest)
            .split_by_rand_pct(0.1, seed=42)
            .label_for_lm()
            .databunch(bs=bs, num_workers=1, backwards=True))

data.save(f'{lang}_databunch_bwd')

# %%
data = load_data(dest, f'{lang}_databunch_bwd', bs=bs, backwards=True)

# %%
learn = language_model_learner(data, AWD_LSTM, drop_mult=0.5, pretrained=False).to_fp16()

# %%
lr = 3e-3
lr *= bs/48  # Scale learning rate by batch size

# %%
learn.unfreeze()
learn.fit_one_cycle(10, lr, moms=(0.8,0.7))

# %%
mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)
learn.to_fp32().save(mdl_path/lm_fns[0], with_opt=False)
learn.data.vocab.save(mdl_path/(lm_fns[1] + '.pkl'))

# %% [markdown]
# ## Vietnamese sentiment analysis

# %% [markdown] heading_collapsed=true
# ### Language model

# %% hidden=true
train_df = pd.read_csv(path/'train.csv')
train_df.loc[pd.isna(train_df.comment),'comment']='NA'

test_df = pd.read_csv(path/'test.csv')
test_df.loc[pd.isna(test_df.comment),'comment']='NA'
test_df['label'] = 0

df = pd.concat([train_df,test_df])

# %% hidden=true
data_lm = (TextList.from_df(df, path, cols='comment')
    .split_by_rand_pct(0.1, seed=42)
    .label_for_lm()           
    .databunch(bs=bs, num_workers=1, backwards=True))

learn_lm = language_model_learner(data_lm, AWD_LSTM, config={**awd_lstm_lm_config, 'n_hid': 1152},
                                  pretrained_fnames=lm_fns, drop_mult=1.0)

# %% hidden=true
lr = 1e-3
lr *= bs/48

# %% hidden=true
learn_lm.fit_one_cycle(2, lr*10, moms=(0.8,0.7))

# %% hidden=true
learn_lm.unfreeze()
learn_lm.fit_one_cycle(8, lr, moms=(0.8,0.7))

# %% hidden=true
learn_lm.save(f'{lang}fine_tuned_bwd')
learn_lm.save_encoder(f'{lang}fine_tuned_enc_bwd')

# %% [markdown]
# ### Classifier

# %%
data_clas = (TextList.from_df(train_df, path, vocab=data_lm.vocab, cols='comment')
    .split_by_rand_pct(0.1, seed=42)
    .label_from_df(cols='label')
    .databunch(bs=bs, num_workers=1, backwards=True))

data_clas.save(f'{lang}_textlist_class_bwd')

# %%
data_clas = load_data(path, f'{lang}_textlist_class_bwd', bs=bs, num_workers=1, backwards=True)

# %%
from sklearn.metrics import f1_score

@np_func
def f1(inp,targ): return f1_score(targ, np.argmax(inp, axis=-1))


# %%
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics=[accuracy,f1]).to_fp16()
learn_c.load_encoder(f'{lang}fine_tuned_enc_bwd')
learn_c.freeze()

# %%
lr=2e-2
lr *= bs/48

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
learn_c.save(f'{lang}clas_bwd')

# %%
