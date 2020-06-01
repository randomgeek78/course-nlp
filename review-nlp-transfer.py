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
# ## Quick Start: Training an IMDb sentiment model with *ULMFiT*

# %% [markdown]
# Let's start with a quick end-to-end example of training a model. We'll train a sentiment classifier on a sample of the popular IMDb data, showing 4 steps:
#
# 1. Reading and viewing the IMDb data
# 1. Getting your data ready for modeling
# 1. Fine-tuning a language model
# 1. Building a classifier

# %%
from fastai.text import * 

# %% [markdown]
# Contrary to images in Computer Vision, text can't directly be transformed into numbers to be fed into a model. The first thing we need to do is to preprocess our data so that we change the raw texts to lists of words, or tokens (a step that is called tokenization) then transform these tokens into numbers (a step that is called numericalization). These numbers are then passed to embedding layers that will convert them in arrays of floats before passing them through a model.
#
# Steps:
#
# 1. Get your data preprocessed and ready to use,
# 1. Create a language model with pretrained weights that you can fine-tune to your dataset,
# 1. Create other models such as classifiers on top of the encoder of the language model.
#
# To show examples, we have provided a small sample of the [IMDB dataset](https://www.imdb.com/interfaces/) which contains 1,000 reviews of movies with labels (positive or negative).

# %%
path = untar_data(URLs.IMDB_SAMPLE)

# %%
df = pd.read_csv(path/'texts.csv')
df.head()

# %% [markdown]
# #### Get the data and create a `databunch` object to use for a language model

# %%
# %%time

# throws `BrokenProcessPool` Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # The following line throws `AttributeError: backwards` on the learning step, below
        # data_lm = TextDataBunch.from_csv(path, 'texts.csv')
        # This Fastai Forum post shows the solution:
        #      https://forums.fast.ai/t/backwards-attributes-not-found-in-nlp-text-learner/51340?u=jcatanza
        # We implement the solution on the following line:
        data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% [markdown]
# #### Get the data again, this time form a `databunch` object for use in a classifier model

# %%
# %%time

# throws `BrokenProcessPool` Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Create the databunch for the classifier model
        data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')


# %% [markdown]
# #### Save the `databunch` objects for later use

# %%
data_lm.save('data_lm_export.pkl')
data_clas.save('data_clas_export.pkl')

# %% [markdown]
# #### Load the `databunch` objects
# Note that you can load the data with different [`DataBunch`](/basic_data.html#DataBunch) parameters (batch size, `bptt`,...)

# %%
#bs=192
#bs=48
bs=32

# %%
data_lm = load_data(path, 'data_lm_export.pkl', bs=bs)
data_clas = load_data(path, 'data_clas_export.pkl', bs=bs)

# %% [markdown]
# ### 1. Build and fine-tune a language model

# %% [markdown]
# We can use the `data_lm` object we created earlier to fine-tune a pretrained language model. [fast.ai](http://www.fast.ai/) has an English model with an AWD-LSTM architecture available that we can download. We can create a learner object that will directly create a model, download the pretrained weights and be ready for fine-tuning.

# %% [markdown]
# #### Set up to use the GPU

# %%
# torch.cuda.set_device(1)
torch.cuda.set_device(0)

# %% [markdown]
# #### Build the IMDb language model, initializing with the hpretrained weights from `wikitext-103`

# %%
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)

# %% [markdown]
# #### Start training
# By default this step unfreezes and trains the weights for the last layer. Training updates the weights to values more applicable to the language of IMDb reviews.

# %%
learn.fit_one_cycle(1, 1e-2)

# %%
# language_model_learner??

# %%
# learn.load_pretrained??

# %%
# convert_weights??

# %% [markdown]
# You can use [Visual Studio Code](https://code.visualstudio.com/) (vscode - open source editor that comes with recent versions of Anaconda, or can be installed separately), or most editors and IDEs, to browse code. vscode things to know:
#
# - Command palette (<kbd>Ctrl-shift-p</kbd>)
# - Go to symbol (<kbd>Ctrl-t</kbd>)
# - Find references (<kbd>Shift-F12</kbd>)
# - Go to definition (<kbd>F12</kbd>)
# - Go back (<kbd>alt-left</kbd>)
# - View documentation
# - Hide sidebar (<kbd>Ctrl-b</kbd>)
# - Zen mode (<kbd>Ctrl-k,z</kbd>)

# %% [markdown]
# #### Unfreeze the weights and train some more

# %% [markdown]
# Like a computer vision model, we can then unfreeze the model and fine-tune it.
# In this step we are allowing the model to update *all* the weights with values more suitable to the language of IMDb reviews. But we are mindful that the pretrained weights from wikitext-103 are likely already near their optimal values. For this we train the weights in the "earlier" layers with a much lower learning rate.

# %%
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4,1e-2))

# %% [markdown]
# #### Test the language model

# %% [markdown]
# To evaluate your language model, you can run the [`Learner.predict`](/basic_train.html#Learner.predict) method and specify the number of words you want it to guess.

# %%
learn.predict("This is a review about", n_words=10)

# %%
learn.predict("In the hierarchy of horror movies this has to be near the top.", n_words=10)

# %% [markdown]
# Sometimes the generated text doesn't make much sense because we have a tiny vocabulary and didn't train much. But note that the model respects basic grammar, which comes from the pretrained model.
#
# Finally we save the encoder to be able to use it for classification in the next section.

# %% [markdown]
# #### Save the encoder

# %%
learn.save('mini_imdb_language_model')
learn.save_encoder('mini_imdb_language_model_encoder')

# %% [markdown]
# ### 2. Build a movie review classifier
# We use mixed precision (`.to_fp16()`)for greater speed, smaller memory footprint, and a regularizing effect.

# %%
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()
learn.load_encoder('mini_imdb_language_model_encoder')

# %%
data_clas.show_batch()

# %% [markdown]
# #### Train the last layer of the classifier

# %%
learn.fit_one_cycle(1, 1e-2)

# %% [markdown]
# #### Unfreeze the classifier model and fine-tune

# %%
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-2))

# %% [markdown]
# #### Test the classifier
# We can use our model to predict on a few example of movie review-like raw text by using the [`Learner.predict`](/basic_train.html#Learner.predict) method. 

# %% [markdown]
# Our model is 70% sure that this is a positive review.

# %%
learn.predict('Although there was lots of blood and violence, I did not think this film was scary enough.')

# %% [markdown]
# Our model is 83% sure that this is a positive review:

# %%
learn.predict('Not so good World War II epic film')

# %% [markdown]
# Bottom line: the model did not do a good job, misclassifying both reviews (with high confidence, to boot!)

# %% [markdown]
# #### Let's train some more, and re-try these examples to see if we can do better

# %%
learn.unfreeze()
learn.fit_one_cycle(3, slice(1e-4, 1e-2))

# %% [markdown]
# #### Re-try the above examples

# %%
learn.predict('Although there was lots of blood and violence, I did not think this film was scary enough.')

# %%
learn.predict('Not so good World War II epic film')

# %% [markdown]
# The model seems to have benefitted from the extra training, since it now correctly classifies both reviews as `negative`, with high confidence.

# %% [markdown]
# #### Make up your own text in the style of a movie review. Then use our classifier to see what it thinks!

# %%
