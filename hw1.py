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
# # Homework 1

# %% [markdown]
# Due on: 5/30.  Please upload your completed assignment to Canvas

# %% [markdown]
# 1. For the following words: logistic, logistics, shoe, shoes
#
#    a. Porter stem with nltk
#    
#    b. lemmatize with nltk
#    
#    c. lemmatize with Spacy

# %% [markdown]
# 2\. n-grams are an important NLP concept.  An n-gram is a contiguous sequence of n items (where the items can be characters, syllables, or words).  Here, we  A 1-gram is a unigram, a 2-gram is a bigram, and a 3-gram is a trigram.
#
# Here, we are referring to sequences of words. The sentence "It was a bright cold day in April." contains the following trigrams:
#
# - It was a
# - was a bright
# - a bright cold
# - bright cold day
# - cold day in
# - day in April
#
# Write a function that returns a dictionary with the n-grams of a text (for `min_n <= n <= max_n`) and a count of how often they appear:

# %%
def get_ngrams(text, min_n, max_n):
    
    #Exercise: FILL IN METHOD
    
    return ngram_dict


# %% [markdown]
# 3\. Write a method that given a list of strings (you can think of each string as a document), returns a dictionary for each string, where the keys are the vocabulary words, and the values are the frequencies.

# %%
def get_vocab_frequency(list_of_strings):

    return list_of_dicts


# %% [markdown]
# 4\. Write a method that when given a list of strings (you can think of each string as a document), calculates the TF-IDF, and returns a term-document matrix with the results. It will be useful to use your `get_vocab_frequency` method from problem 3.

# %%
def get_tfidf(list_of_strings):
    
    
    return tfidf

# %% [markdown]
# 5\. Who said the following (*Hint: Be sure to read the class notebooks and relevant links*):
#
# A. "It's true there's been a lot of work on trying to apply statistical models to various linguistic problems. I think there have been some successes, but a lot of failures. There is a notion of success ... which I think is novel in the history of science. It interprets success as approximating unanalyzed data."
#
# B. "I agree that it can be difficult to make sense of a model containing billions of parameters. Certainly a human can't understand such a model by inspecting the values of each parameter individually. But one can gain insight by examing the properties of the model—where it succeeds and fails, how well it learns as a function of data, etc."
#
# C. The big-data big-compute paradigm of modern Deep Learning has in fact “perverted the field” (of computational linguistics) and “sent it off-track”
#
# D. Language is crucial to general intelligence, because language is the conduit by which individual intelligence is shared and transformed into societal intelligence.
#
# E. Structure is a “necessary evil”, and warned that imposing structure requires us to make certain assumptions, which are invariably wrong for at least some portion of the data, and may become obsolete within the near future.

# %%
