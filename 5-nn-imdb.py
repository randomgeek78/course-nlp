# -*- coding: utf-8 -*-
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
# This lesson was adapted from the end of [lesson 3](https://course.fast.ai/videos/?lesson=3) and beginning of [lesson 4](https://course.fast.ai/videos/?lesson=4) of the latest fast.ai Practical Deep Learning for Coders course.  We will cover all the material you need here in this notebook, so no need to have taken the Deep Learning course.  Even if you have taken the DL class, we will go slower and get into more detail here!

# %% [markdown]
# # Transfer Learning for Natural Language Modeling
# ### Contructing a Language Model and a Sentiment Classifier for IMDB movie reviews

# %% [markdown]
# Transfer learning has been widely used with great success in computer vision for several years, but only in the last year or so has it been successfully applied to NLP (beginning with ULMFit, which we will use here, which was built upon by BERT and GPT-2).
#
# As Sebastian Ruder wrote in [The Gradient](https://thegradient.pub/) last summer, [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/).
#
# We will first build a language model for IMDB movie reviews.  Next we will build a sentiment classifier, which will predict whether a review is negative or positive, based on its text. For both of these tasks, we will use **transfer learning**. Starting with the pre-trained weights from the `wikitext-103` language model, we will tune these weights to specialize to the language of `IMDb` movie reviews. 

# %% [markdown] heading_collapsed=true
# ## Language Models

# %% [markdown] hidden=true
# Language modeling can be a fun creative form. Research scientist [Janelle Shane blogs](https://aiweirdness.com/) & [tweets](https://twitter.com/JanelleCShane) about her creative AI explorations, which often involve text.  For instance, see her:
#
# - [Why did the neural network cross the road?](https://aiweirdness.com/post/174691534037/why-did-the-neural-network-cross-the-road)
# - [Try these neural network-generated recipes at your own risk.](https://aiweirdness.com/post/163878889437/try-these-neural-network-generated-recipes-at-your)
# - [D&D character bios - now making slightly more sense](https://aiweirdness.com/post/183471928977/dd-character-bios-now-making-slightly-more)

# %% [markdown]
# ## Using a GPU

# %% [markdown]
# You will need to have the fastai library installed for this lesson, and you will want to use a GPU to train your neural net.  If you don't have a GPU you can use in your computer (currently, only Nvidia GPUs are fully supported by the main deep learning libraries), no worries!  There are a number of cloud options you can consider:
#
# [GPU Cloud Options](https://course.fast.ai/#using-a-gpu)
#
# **Reminder: If you are using a cloud GPU, always be sure to shut it down when you are done!!! Otherwise, you could end up with an expensive bill!**

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
from fastai import *
from fastai.text import *
from scipy.spatial.distance import cosine as dist

# %% [markdown]
# Note that language models can use a lot of GPU, so you may need to decrease batchsize here.

# %%
# bs=192
bs=48
# bs=24

# %% [markdown]
# ### Fix this line: should be `device(0)` instead of `device(2)`

# %%
#torch.cuda.set_device(2)
torch.cuda.set_device(0)

# %% [markdown] heading_collapsed=true
# ## 1. Prepare the IMDb data (on a sample)

# %% [markdown] hidden=true
# First let's download the dataset we are going to study. The `IMDb` [dataset](http://ai.stanford.edu/~amaas/data/sentiment/) has been curated by Andrew Maas et al. and contains a total of 100,000 reviews on IMDB. 25,000 of them are labelled as positive and negative for training, another 25,000 are labelled for testing (in both cases they are highly polarized). The remaning 50,000 is an additional unlabelled data (but we will find a use for it nonetheless).
#
# We'll begin with a sample we've prepared for you, so that things run quickly before going over the full dataset.

# %% hidden=true
path = untar_data(URLs.IMDB_SAMPLE)
path.ls()

# %% [markdown] hidden=true
# It only contains one csv file, let's have a look at it.

# %% [markdown] hidden=true
# It contains one line per review, with the label ('negative' or 'positive'), the text and a flag to determine if it should be part of the validation set or the training set. If we ignore this flag, we can create a `DataBunch` containing this data in one line of code:

# %% [markdown]
# ### Load and preprocess the data and form a `databunch`
# Add workaround for the bug in the `fastai Text API`

# %% hidden=true
# %%time

# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        data_lm = TextDataBunch.from_csv(path, 'texts.csv')
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% [markdown] hidden=true
# By executing this line a process was launched that took a bit of time. Let's dig a bit into it. Images could be fed (almost) directly into a model because they're just a big array of pixel values that are floats between 0 and 1. A text is composed of words, and we can't apply mathematical functions to them directly. We first have to convert them to numbers. This is done in two differents steps: tokenization and numericalization. A `TextDataBunch` does all of that behind the scenes for you.

# %% [markdown] heading_collapsed=true hidden=true
# ### Tokenization

# %% [markdown] hidden=true
# The first step of processing we make texts go through is to split the raw sentences into words, or more exactly tokens. The easiest way to do this would be to split the string on spaces, but we can be smarter:
#
# - we need to take care of punctuation
# - some words are contractions of two different words, like isn't or don't
# - we may need to clean some parts of our texts, if there's HTML code for instance
#
# To see what the tokenizer had done behind the scenes, let's have a look at a few texts in a batch.

# %% [markdown] hidden=true
# The texts are truncated at 100 tokens for more readability. We can see that it did more than just split on space and punctuation symbols: 
# - the "'s" are grouped together in one token
# - the contractions are separated like his: "did", "n't"
# - content has been cleaned for any HTML symbol and lower cased
# - there are several special tokens (all those that begin by xx), to replace unkown tokens (see below) or to introduce different text fields (here we only have one).

# %% [markdown] heading_collapsed=true hidden=true
# ### Numericalization

# %% [markdown] hidden=true
# Once we have extracted tokens from our texts, we convert to integers by creating a list of all the words used. We only keep the ones that appear at list twice with a maximum vocabulary size of 60,000 (by default) and replace the ones that don't make the cut by the unknown token `UNK`.
#
# The correspondance from ids tokens is stored in the `vocab` attribute of our datasets, in a dictionary called `itos` (for int to string).

# %% hidden=true
data_lm.vocab.itos[:10]

# %% [markdown] hidden=true
# And if we look at what a what's in our datasets, we'll see the tokenized text as a representation:

# %% hidden=true
data_lm.train_ds[0][0]

# %% [markdown] hidden=true
# But the underlying data is all numbers

# %% hidden=true
data_lm.train_ds[0][0].data[:10]

# %% [markdown] heading_collapsed=true hidden=true
# ### Alternative approach: with the `data block API`

# %% [markdown] hidden=true
# We can use the data block API with NLP and have a lot more flexibility than what the default factory methods offer. In the previous example for instance, the data was randomly split between train and validation instead of reading the third column of the csv.
#
# With the data block API though, we have to manually call the tokenize and numericalize steps. This allows more flexibility, and if you're not using the defaults from fastai, the various arguments to pass will appear in the step they're revelant, so it'll be more readable.

# %% [markdown]
# ### Load and preprocess the data and form a `datablock`
# Add workaround for the bug in the `fastai Text API`

# %% hidden=true
# %%time

# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        data = (TextList.from_csv(path, 'texts.csv', cols='text')
                .split_from_df(col=2)
                .label_from_df(cols=0)
                .databunch())        
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% [markdown]
# ## 2. Transfer Learning <br>
# ### We are going to create an `IMDb` language model starting with the pretrained weights from the `wikitext-103` language model.

# %% [markdown]
# Now let's grab the full `IMDb` dataset for what follows.

# %%
path = untar_data(URLs.IMDB)
path.ls()

# %%
(path/'train').ls()

# %% [markdown]
# The reviews are in a training and test set following an imagenet structure. The only difference is that there is an `unsup` folder in `train` that contains the unlabelled data.
#
# We're not going to train a model that classifies the reviews from scratch. Like in computer vision, we'll use a model pretrained on a bigger dataset (a cleaned subset of wikipedia called [wikitext-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)). That model has been trained to guess what the next word, its input being all the previous words. It has a recurrent structure and a hidden state that is updated each time it sees a new word. This hidden state thus contains information about the sentence up to that point.
#
# We are going to use that 'knowledge' of the English language to build our classifier, but first, like for computer vision, we need to fine-tune the pretrained model to our particular dataset. Because the English of the reviews left by people on IMDB isn't the same as the English of wikipedia, we'll need to adjust a little bit the parameters of our model. Plus there might be some words extremely common in that dataset that were barely present in wikipedia, and therefore might no be part of the vocabulary the model was trained on.

# %% [markdown] heading_collapsed=true
# ### More about WikiText-103

# %% [markdown] hidden=true
# We will be using the `WikiText-103` dataset created by [Stephen Merity](https://smerity.com/) to pre-train a language model.
#
# To quote [Stephen's post](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/):
#
# *The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.*
#
# *Compared to the preprocessed version of Penn Treebank (PTB), WikiText-2 is over 2 times larger and WikiText-103 is over 110 times larger. The WikiText dataset also features a far larger vocabulary and retains the original case, punctuation and numbers - all of which are removed in PTB. As it is composed of full articles, the dataset is well suited for models that can take advantage of long term dependencies.*

# %% [markdown]
# [Download wikitext-103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip). Unzip it into the `.fastai/data/` folder on your computer.

# %% [markdown] heading_collapsed=true
# ### 2A. Package the `IMDb` data into a language model `databunch`

# %% [markdown] hidden=true
# This is where the unlabelled data is going to be useful to us, as we can use it to fine-tune our model. Let's create our data object with the data block API (this takes a few minutes).
#
# We'll to use a special kind of `TextDataBunch` for the language model, that ignores the labels (that's why we put 0 everywhere), will shuffle the texts at each epoch before concatenating them all together (only for training; we don't shuffle for the validation set) and will send batches that read that text in order with targets that are the next word in the sentence.
#
# Add a `try-except` wrapper as a workaround for the bug in the `fastai Text API`

# %% hidden=true
# %%time

# throws `BrokenProcessPool` Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        data_lm = (TextList.from_folder(path)
           #Inputs: all the text files in path
            .filter_by_folder(include=['train', 'test', 'unsup']) 
           # notebook 3-logreg-nb-imbd used .split_by_folder instead of .filter_by_folder
            # and this took less time to run. Can we do the same here?
           #We may have other temp folders that contain text files so we only keep what's in train and test
            .split_by_rand_pct(0.1, seed=42))
           #We randomly split and keep 10% (10,000 reviews) for validation
            #.label_for_lm()           
           #We want to make a language model so we label accordingly
            #.databunch(bs=bs, num_workers=1))
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% [markdown]
# #### I got faster results when I do the last two steps in a separate cell:

# %%
# %%time

# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        #     the next step is the bottleneck
        data_lm = (data_lm.label_for_lm()           
           #We want to make a language model so we label accordingly
            .databunch(bs=bs, num_workers=1))
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% hidden=true
len(data_lm.vocab.itos),len(data_lm.train_ds)

# %% hidden=true
data_lm.show_batch()

# %% [markdown] hidden=true
# #### Save the `databunch` for next time.

# %%
data_lm.save()

# %% [markdown]
# #### Load the saved data

# %%
data_lm = load_data(path, 'lm_databunch', bs=bs)

# %% [markdown]
# ### 2B. The **Transfer Learning** step.
# #### This is where the magic happens!
# #### The `AWD_LSTM` object contains the pretrained weights and the neural net architecture of the `wikitext-103` language model. These will be downloaded the first time you execute the following line, and stored in `~/.fastai/models/` (or elsewhere if you specified different paths in your config file). 
#
# We import these into the `language_model_learner` object for our `IMDb` language model as follows:

# %%
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)

# %% [markdown]
# #### Get the `IMDb` language model `vocabulary`

# %%
vocab = data_lm.vocab

# %%
vocab.stoi["stingray"]

# %%
vocab.itos[vocab.stoi["stingray"]]

# %%
vocab.itos[vocab.stoi["mobula"]]

# %%
awd = learn_lm.model[0]

# %% [markdown]
# #### Get the `IMDb` language model `encoder`. Recall that the `encoder` translates tokens into numerical vectors in the space defined by the `IMDb` vocabulary.

# %%
enc = learn_lm.model[0].encoder

# %%
enc.weight.size()

# %% [markdown] heading_collapsed=true
# #### Difference in vocabulary between IMDB and Wikipedia language models

# %% [markdown] hidden=true
# We are going to load `wiki_itos` (the index-to-string list) from the `wikitext 103` language model.  We will compare the vocabularies of `wikitext-103` and `IMDB`.  It is to be expected that the two sets have some different vocabulary words, and that is no problem for transfer learning!

# %%
#wiki_itos = pickle.load(open(Config().model_path()/'wt103-1/itos_wt103.pkl', 'rb'))
wiki_itos = pickle.load(open(Config().model_path()/'wt103-fwd/itos_wt103.pkl', 'rb'))

# %%
wiki_itos[:10]

# %% hidden=true
len(wiki_itos)

# %% hidden=true
len(vocab.itos)

# %% hidden=true
i, unks = 0, []
while len(unks) < 50:
    if data_lm.vocab.itos[i] not in wiki_itos: unks.append((i,data_lm.vocab.itos[i]))
    i += 1

# %% hidden=true
wiki_words = set(wiki_itos)

# %% hidden=true
imdb_words = set(vocab.itos)

# %% hidden=true
wiki_not_imbdb = wiki_words.difference(imdb_words)

# %% hidden=true
imdb_not_wiki = imdb_words.difference(wiki_words)

# %% hidden=true
wiki_not_imdb_list = []

for i in range(100):
    word = wiki_not_imbdb.pop()
    wiki_not_imdb_list.append(word)
    wiki_not_imbdb.add(word)

# %% hidden=true
wiki_not_imdb_list[:15]

# %% hidden=true
imdb_not_wiki_list = []

for i in range(100):
    word = imdb_not_wiki.pop()
    imdb_not_wiki_list.append(word)
    imdb_not_wiki.add(word)

# %% hidden=true
imdb_not_wiki_list[:15]

# %% [markdown] hidden=true
# All words that appear in the `IMDB` vocab, but not the `wikitext-103` vocab, will be initialized to the same random vector in a model.  As the model trains, we will learn their weights.

# %% hidden=true
vocab.stoi["modernisation"]

# %% hidden=true
"modernisation" in wiki_words

# %% hidden=true
vocab.stoi["30-something"]

# %% hidden=true
"30-something" in wiki_words, "30-something" in imdb_words

# %% hidden=true
vocab.stoi["linklater"]

# %% hidden=true
"linklater" in wiki_words, "linklater" in imdb_words

# %% hidden=true
"house" in wiki_words, "house" in imdb_words

# %% hidden=true
np.allclose(enc.weight[vocab.stoi["30-something"], :], 
            enc.weight[vocab.stoi["linklater"], :])

# %% hidden=true
np.allclose(enc.weight[vocab.stoi["30-something"], :], 
            enc.weight[vocab.stoi["house"], :])

# %% hidden=true
new_word_vec = enc.weight[vocab.stoi["linklater"], :]

# %% [markdown] heading_collapsed=true
# #### Generating fake movie review-like text with the **untrained** `IMDb` language model

# %% hidden=true
TEXT = "The color of the sky is"
N_WORDS = 40
N_SENTENCES = 2

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% hidden=true
TEXT = "I hated this movie"
N_WORDS = 30
N_SENTENCES = 2

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% hidden=true
doc(LanguageLearner.predict)

# %% [markdown] hidden=true
# Lowering the `temperature` will make the texts less randomized.

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.10) for _ in range(N_SENTENCES)))

# %% hidden=true
doc(LanguageLearner.predict)

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.10) for _ in range(N_SENTENCES)))

# %% [markdown]
# ### 2C. Training (fine-tuning) the `IMDb` language model
# #### Starting with the `wikitext-103` pretrained weights, we'll fine-tune the model to "learn" the structure in the "language" of IMDb movie reviews.

# %% [markdown]
# #### Choose an appropriate learning rate.

# %%
learn_lm.lr_find()

# %%
learn_lm.recorder.plot(skip_end=15)

# %%
lr = 1e-3
lr *= bs/48

# %% [markdown]
# #### Use the mixed-precision option, if you have it, otherwise omit this step

# %%
learn_lm.to_fp16()

# %% [markdown]
# #### The first step in fine-tuning is to train only the last layer of the model. 
# This takes about a half-hour on an NVIDIA RTX-2070 GPU

# %%
learn_lm.fit_one_cycle(1, lr*10, moms=(0.8,0.7))

# %% [markdown]
# Since this is relatively slow to train, we will save our weights:

# %%
learn_lm.save('fit_1')

# %%
learn_lm.load('fit_1')

# %% [markdown]
# #### To complete the fine-tuning, we unfreeze all the weights and retrain
# Adopting the `wikitext-103` weights as initial values, our neural network will adjust them via optimization, finding new values that are specialized to the "language" of `IMDb` movie reviews.

# %%
learn_lm.unfreeze()

# %% [markdown]
# Fine tuning the model takes ~30 minutes per epoch on an NVIDIA RTX-2070 GPU, with bs=48<br>
# Note the relatively low value of accuracy, which did not improve significantly beyond `epoch 4`.

# %%
learn_lm.fit_one_cycle(10, lr, moms=(0.8,0.7))

# %% [markdown]
# #### Save the fine-tuned **language model** and the **encoder**
# We have to save not just the `fine-tuned` **IMDb language model** but also its **encoder**. The **language model** is the part that tries to guess the next word. The **encoder** is the part that's responsible for creating and updating the hidden state. 
#
# In the next part we will build a **sentiment classifier** for the IMDb movie reviews. To do this we will need the **encoder** from the **IMDb language model** that we built.

# %%
learn_lm.save('fine_tuned')

# %%
learn_lm.save_encoder('fine_tuned_enc')

# %% [markdown] heading_collapsed=true
# #### Load the saved **model** and its **encoder**

# %% hidden=true
learn_lm.load('fine_tuned')

# %% [markdown] hidden=true
# Now that we've trained our model, different representations have been learned for the words that were in `IMDb` but not `wikitext-103` (remember that at the beginning we had initialized them all to the same thing):

# %% hidden=true
enc = learn_lm.model[0].encoder

# %% hidden=true
np.allclose(enc.weight[vocab.stoi["30-something"], :], 
            enc.weight[vocab.stoi["linklater"], :])

# %% hidden=true
np.allclose(enc.weight[vocab.stoi["30-something"], :], new_word_vec)

# %% [markdown] heading_collapsed=true
# #### Generate movie review-like text, with the **fine-tuned** ` IMDb` language model
# Compare these texts to the ones generated with the **untrained** `IMDb model` in part **2A**. Do they seem qualitatively better?

# %% [markdown] hidden=true
# How good is our fine-tuned IMDb language model? Well let's try to see what it predicts when given a phrase that might appear in a movie review.

# %% hidden=true
TEXT = "i liked this movie because"
N_WORDS = 40
N_SENTENCES = 2

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% hidden=true
TEXT = "This movie was"
N_WORDS = 30
N_SENTENCES = 2

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% hidden=true
TEXT = "I hated this movie"
N_WORDS = 40
N_SENTENCES = 2

# %% hidden=true
print("\n".join(learn_lm.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# %% [markdown] heading_collapsed=true
# #### Risks of language models

# %% [markdown] hidden=true
# We will talk about ethical concerns raised by very accurate language models in lesson 7, but here are a few brief notes:
#
# In reference to [OpenAI's GPT-2](https://www.theverge.com/2019/2/14/18224704/ai-machine-learning-language-models-read-write-openai-gpt2): Jeremy Howard said, *I’ve been trying to warn people about this for a while. We have the technology to totally fill Twitter, email, and the web up with reasonable-sounding, context-appropriate prose, which would drown out all other speech and be impossible to filter.*
#
# For a small example, consider when completely incorrect (but reasonable sounding) ML generated answers were [posted to StackOverflow](https://meta.stackoverflow.com/questions/384596/completely-incorrect-machine-learning-generated-answers?stw=2):
#
# <img src="images/robot-overflow.png" alt="Roboflow" style="width: 80%"/>

# %% [markdown] hidden=true
# <img src="images/husain-tweet.png" alt="Roboflow" style="width: 60%"/>

# %% [markdown]
# ## 3. Building an `IMDb Sentiment Classifier`
# #### We'll now use **transfer learning** to create a `classifier`, again starting from the pretrained weights of the `wikitext-103` language model.  We'll also need the `IMDb language model` **encoder** that we saved previously. 

# %% [markdown]
# ### 3A. Load and preprocess the data, and form a `databunch`
# Using fastai's flexible API, we will now create a different kind of `databunch` object, one that is suitable for a **classifier** rather than a for **language model** (as we did in **2A**). This time we'll keep the labels for the `IMDb` movie reviews data. 
#
# Add the `try-except` wrapper workaround for the bug in the `fastai Text API`
#
# Here the batch size is decreased from 48 to 8, to avoid a `CUDA out of memory error`; your hardware may be able to handle a larger batch, in which case training will likely be faster.
#
# Again, this takes a bit of time.

# %%
bs=8

# %% hidden=true
# %%time

# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!
#    the progress bar has to complete three consecutive steps. Why three? 
#    fails nearly 100 times, and doesn't respond to interrupt
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        data_clas = (TextList.from_folder(path, vocab=data_lm.vocab)
             #grab all the text files in path
             .split_by_folder(valid='test')
             #split by train and valid folder (that only keeps 'train' and 'test' so no need to filter)
             .label_from_folder(classes=['neg', 'pos']))
             #label them all with their folders
             #.databunch(bs=bs, num_workers=1))
        error = False
        print(f'failure count is {count}\n')   
        
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')

# %% [markdown]
# #### Form the preprocessed data into a `databunch`

# %%
data_clas = data_clas.databunch(bs=bs, num_workers=1)

# %% [markdown]
# #### Save the databunch (since it took so long to make) and load it

# %%
data_clas.save('imdb_textlist_class')

# %%
data_clas = load_data(path, 'imdb_textlist_class', bs=bs, num_workers=1)

# %%
data_clas.show_batch()

# %% [markdown]
# ### 3B. Create a model to **classify** the `IMDb` reviews, and load the **encoder** we saved before.
# #### Freeze the weights for all but the last layer and find a good value for the learning rate. 

# %%
learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3).to_fp16()
learn_c.load_encoder('fine_tuned_enc')
learn_c.freeze()

# %%
learn_c.lr_find()

# %%
learn_c.recorder.plot()

# %% [markdown]
# ### 3C. Training and fine-tuning the `IMDb sentiment classifier`

# %% [markdown]
# #### Train for one cycle, save intermediate result

# %%
learn_c.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

# %%
learn_c.save('first')

# %%
learn_c.load('first');

# %% [markdown]
# #### Unfreeze last two layers and train for one cycle, save intermediate result.

# %%
learn_c.freeze_to(-2)
learn_c.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

# %%
learn_c.save('2nd')

# %% [markdown]
# #### Unfreeze the last three layers, and train for one cycle, and save intermediate result.
# At this point we've already beaten the 2017 (pre-transfer learning) state of the art!

# %%
learn_c.freeze_to(-3)
learn_c.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

# %%
learn_c.save('3rd')

# %%
learn_c.load('3rd')

# %% [markdown]
# #### Unfreeze all the layers, train for two cycles, and save the result.

# %% [markdown]
# Note: at this step I encountered a `CUDA error: unspecified launch failure`. This is a known (and unsolved) problem with PyTorch when using an LSTM. https://github.com/pytorch/pytorch/issues/27837
#
# Nothing to do but try again... and it worked on the second try.

# %%
learn_c.unfreeze()
learn_c.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

# %% [markdown]
# The state of the art for this dataset in 2017 was 94.1%, and we have crushed it!!!

# %% [markdown]
# #### Save the IMDb classifer model

# %%
learn_c.save('clas')

# %% [markdown]
# #### Let's look at a few examples, just to check that the classifier is working as we think it should. 
# The three outputs of the model predition are the label (`pos` or `neg`) and the class probability estimates for `neg` and `pos`, which meausure the model's confidence in it's prediction. As we'd expect, the model is extremely confident that the first review is `pos` and quite confident that the second review is `neg`. So it passes the test with flying colors. 

# %%
learn_c.predict("I really loved that movie, it was awesome!")

# %%
learn_c.predict("I didn't really love that movie, and I didn't think it was awesome.")

# %% [markdown]
# #### Now that we've built the model, here is the part where you get to have some fun!! Take the model for a spin, try out your own examples!!

# %% [markdown] heading_collapsed=true
# ## Appendix: Language Model Zoo

# %% [markdown] hidden=true
# fast.ai alumni have applied ULMFit to dozens of different languages, and have beat the SOTA in Thai, Polish, German, Indonesian, Hindi, & Malay.
#
# They share tips and best practices in [this forum thread](https://forums.fast.ai/t/language-model-zoo-gorilla/14623) in case you are interested in getting involved!
#
# <img src="images/language_model_zoo.png" alt="language model zoo" style="width: 80%"/>

# %% hidden=true
