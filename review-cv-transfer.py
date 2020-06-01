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
# # Review: computer vision transfer learning

# %% [markdown]
# This is a subset of lesson 1 of https://course.fast.ai

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
from fastai.metrics import error_rate

# %%
bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

# %% [markdown]
# We are going to use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) by [O. M. Parkhi et al., 2012](http://www.robots.ox.ac.uk/~vgg/publications/2012/parkhi12a/parkhi12a.pdf) which features 12 cat breeds and 25 dogs breeds. Our model will need to learn to differentiate between these 37 distinct categories. According to their paper, the best accuracy they could get in 2012 was 59.21%.

# %%
path = untar_data(URLs.PETS)
path_anno = path/'annotations'
path_img = path/'images'
np.random.seed(2)
fnames = get_image_files(path_img)
fnames[0]

# %%
pat = r'/([^/]+)_\d+.jpg$'
data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))

# %% [markdown]
# ## Training

# %%
learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# %%
learn.fit_one_cycle(4)

# %%
learn.save('stage-1')

# %%
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

# %%
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)

# %%
