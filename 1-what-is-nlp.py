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
# These notebooks are found in the repo: https://github.com/fastai/course-nlp

# %% [markdown]
# # 1. What is NLP?

# %% [markdown]
# ## What can you do with NLP?

# %% [markdown]
# NLP is a broad field, encompassing a variety of tasks, including:
#
# - Part-of-speech tagging: identify if each word is a noun, verb, adjective, etc.)
# - Named entity recognition NER): identify person names, organizations, locations, medical codes, time expressions, quantities, monetary values, etc)
# - Question answering
# - Speech recognition
# - Text-to-speech and Speech-to-text
# - Topic modeling
# - Sentiment classification
# - Language modeling
# - Translation

# %% [markdown]
# Many techniques from NLP are useful in a variety of places, for instance, you may have text within your tabular data.

# %% [markdown]
# There are also interesting techniques that let you go between text and images:
#
# <img src="images/tench-net.png" alt="" style="width: 65%"/>
#
# from [Lesson 9](http://course18.fast.ai/lessons/lesson11.html) of Practical Deep Learning for Coders 2018.

# %% [markdown]
# ## This course

# %% [markdown]
# In this course, we will cover these applications:
# - Topic modeling
# - Sentiment classification
# - Language modeling
# - Translation

# %% [markdown]
# ### Top-down teaching approach

# %% [markdown]
# I'll be using a *top-down* teaching method, which is different from how most CS/math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.
#
# If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [in this talk](https://vimeo.com/214233053).
#
# Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.
#
# All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" that haven't yet been explained, and then we'll dig into the lower level details later. The goal is to get experience working with interesting applications, which will motivate you to learn more about the underlying structures as time goes on.
#
# To start, focus on what things DO, not what they ARE.

# %% [markdown]
# ## A changing field

# %% [markdown]
# Historically, NLP originally relied on hard-coded rules about a language.  In the 1990s, there was a change towards using statistical & machine learning approaches, but the complexity of natural language meant that simple statistical approaches were often not state-of-the-art. We are now currently in the midst of a major change in the move towards neural networks.  Because deep learning allows for much greater complexity, it is now achieving state-of-the-art for many things.
#
# This doesn't have to be binary: there is room to combine deep learning with rules-based approaches.

# %% [markdown]
# ### Case study: spell checkers

# %% [markdown]
# This example comes from Peter Norvig: 

# %% [markdown]
# Historically, spell checkers required thousands of lines of hard-coded rules:
#
# <img src="images/spellchecker1.png" alt="" style="width: 60%"/>

# %% [markdown]
# A version that uses historical data and probabilities can be written in far fewer lines:
#
# <img src="images/spellchecker2.png" alt="" style="width: 60%"/>
#
# [Read more here](http://norvig.com/spell-correct.html). 

# %% [markdown] heading_collapsed=true
# ### A field in flux

# %% [markdown] hidden=true
# The field is still very much in a state of flux, with best practices changing.

# %% [markdown] hidden=true
# <img src="images/skomoroch.png" alt="" style="width: 60%"/>

# %% [markdown]
# ### Norvig vs. Chomsky

# %% [markdown]
# This "debate" captures the tension between two approaches:
#
# - modeling the underlying mechanism of a phenomena
# - using machine learning to predict outputs (without necessarily understanding the mechanisms that create them)
#
# This tension is still very much present in NLP (and in many fields in which machine learning is being adopted, as well as in approachs to "artificial intelligence" in general).

# %% [markdown]
# Background: Noam Chomsky is an MIT emeritus professor, the father of modern linguistics, one of the founders of cognitive science, has written >100 books. Peter Norvig is Director of Research at Google.

# %% [markdown]
# From [MIT Tech Review coverage](https://www.technologyreview.com/s/423917/unthinking-machines/) of a panel at MIT in 2011:
#
# "Chomsky derided researchers in machine learning who use purely statistical methods to produce behavior that mimics something in the world, but who don’t try to understand the meaning of that behavior. Chomsky compared such researchers to scientists who might study the dance made by a bee returning to the hive, and who could produce a statistically based simulation of such a dance without attempting to understand why the bee behaved that way. “That’s a notion of scientific success that’s very novel. I don’t know of anything like it in the history of science,” said Chomsky."

# %% [markdown]
# From Norvig's response [On Chomsky and the Two Cultures of Statistical Learning](http://norvig.com/chomsky.html):
#
# "Breiman is inviting us to give up on the idea that we can uniquely model the true underlying form of nature's function from inputs to outputs. Instead he asks us to be satisfied with a function that accounts for the observed data well, and generalizes to new, previously unseen data well, but may be expressed in a complex mathematical form that may bear no relation to the "true" function's form (if such a true function even exists). Chomsky takes the opposite approach: he prefers to keep a simple, elegant model, and give up on the idea that the model will represent the data well."

# %% [markdown]
# - [Noam Chomsky on Where Artificial Intelligence Went Wrong: An extended conversation with the legendary linguist](https://www.theatlantic.com/technology/archive/2012/11/noam-chomsky-on-where-artificial-intelligence-went-wrong/261637/)
# - [Norvig vs. Chomsky and the Fight for the Future of AI](https://www.tor.com/2011/06/21/norvig-vs-chomsky-and-the-fight-for-the-future-of-ai/)

# %% [markdown]
# ### Yann LeCun vs. Chris Manning

# %% [markdown]
# Another interesting discussion along the topic of how much linguistic structure to incorporate into NLP models is between Yann LeCun and Chris Manning:

# %% [markdown]
# [Deep Learning, Structure and Innate Priors: A Discussion between Yann LeCun and Christopher Manning](http://www.abigailsee.com/2018/02/21/deep-learning-structure-and-innate-priors.html):
#
# *On one side, Manning is a prominent advocate for incorporating more linguistic structure into deep learning systems. On the other, LeCun is a leading proponent for the ability of simple but powerful neural architectures to perform sophisticated tasks without extensive task-specific feature engineering. For this reason, anticipation for disagreement between the two was high, with one Twitter commentator describing the event as “the AI equivalent of Batman vs Superman”.*
#
# ...
#
# *Manning described structure as a “necessary good” (9:14), arguing that we should have a positive attitude towards structure as a good design decision. In particular, structure allows us to design systems that can learn more from less data, and at a higher level of abstraction, compared to those without structure.*
#
# *Conversely, LeCun described structure as a “necessary evil” (2:44), and warned that imposing structure requires us to make certain assumptions, which are invariably wrong for at least some portion of the data, and may become obsolete within the near future. As an example, he hypothesized that ConvNets may be obsolete in 10 years (29:57).*

# %% [markdown]
# ## Resources

# %% [markdown]
# **Books**
#
# We won't be using a text book, although here are a few helpful references:
#
# - [**Speech and Language Processing**](https://web.stanford.edu/~jurafsky/slp3/), by Dan Jurafsky and James H. Martin (free PDF)
#
# - [**Introduction to Information Retrieval**](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) by By Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze (free online)
#
# - [**Natural Language Processing with PyTorch**](https://learning.oreilly.com/library/view/natural-language-processing/9781491978221/) by Brian McMahan and Delip Rao (need to purchase or have O'Reilly Safari account) 
#
# **Blogs**
#
# Good NLP-related blogs:
# - [Sebastian Ruder](http://ruder.io/)
# - [Joyce Xu](https://medium.com/@joycex99)
# - [Jay Alammar](https://jalammar.github.io/)
# - [Stephen Merity](https://smerity.com/articles/articles.html)
# - [Rachael Tatman](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)

# %% [markdown]
# ## NLP Tools

# %% [markdown]
# - Regex (example: find all phone numbers: 123-456-7890, (123) 456-7890, etc.)
# - Tokenization: splitting your text into meaningful units (has a different meaning in security)
# - Word embeddings
# - Linear algebra/matrix decomposition
# - Neural nets
# - Hidden Markov Models
# - Parse trees
#
# Example from http://damir.cavar.me/charty/python/: "She killed the man with the tie."
#
# Was the man wearing the tie?
# <img src="images/parse2.png" alt="" style="width: 60%"/>
#
# Or was the tie the murder weapon?
# <img src="images/parse1.png" alt="" style="width: 60%"/>

# %% [markdown]
# ## Python Libraries

# %% [markdown]
# - [nltk](https://www.nltk.org/): first released in 2001, very broad NLP library
# - [spaCy](https://spacy.io/): creates parse trees, excellent tokenizer, opinionated
# - [gensim](https://radimrehurek.com/gensim/): topic modeling and similarity detection
#
# specialized tools:
# - [PyText](https://pytext-pytext.readthedocs-hosted.com/en/latest/)
# - [fastText](https://fasttext.cc/) has library of embeddings
#
# general ML/DL libraries with text features:
# - [sklearn](https://scikit-learn.org/stable/): general purpose Python ML library
# - [fastai](https://docs.fast.ai/): fast & accurate neural nets using modern best practices, on top of PyTorch

# %% [markdown]
# ## A few NLP applications from fast.ai students

# %% [markdown]
# Some things you can do with NLP:
#
# - [How Quid uses deep learning with small data](https://quid.com/feed/how-quid-uses-deep-learning-with-small-data): Quid has a database of company descriptions, and needed to identify company descriptions that are low quality (too much generic marketing language)
#
# <img src="images/quid1.png" alt="" style="width: 65%"/>

# %% [markdown]
# - Legal Text Classifier with Universal Language Model Fine-Tuning: A law student in Singapore classified legal documents by category (civil, criminal, contract, family, tort,...)
#
# <img src="images/lim-how-kang-1.png" alt="" style="width: 65%"/>
# <img src="images/lim-how-kang-2.png" alt="" style="width: 65%"/>

# %% [markdown]
# - [Democrats ‘went low’ on Twitter leading up to 2018](https://www.rollcall.com/news/campaigns/lead-midterms-twitter-republicans-went-high-democrats-went-low): Journalism grad students analyzed twitter sentiment of politicians
#
# <img src="images/floris-wu.png" alt="" style="width: 65%"/>

# %% [markdown]
# - [Introducing Metadata Enhanced ULMFiT](https://www.novetta.com/2019/03/introducing_me_ulmfit/), classifying quotes from articles.  Uses metadata (such as publication, country, and source) together with the text of the quote to improve accuracy of the classifier.
#
# <img src="images/novetta1.png" alt="" style="width: 65%"/>

# %% [markdown]
# ## Ethics issues

# %% [markdown]
# ### Bias

# %% [markdown]
# <img src="images/google-translate.png" alt="" style="width: 65%"/>

# %% [markdown]
# - [How Vector Space Mathematics Reveals the Hidden Sexism in Language](https://www.technologyreview.com/s/602025/how-vector-space-mathematics-reveals-the-hidden-sexism-in-language/)
# - [Semantics derived automatically from language corpora contain human-like biases](https://arxiv.org/abs/1608.07187)
# - [Lipstick on a Pig: Debiasing Methods Cover up Systematic Gender Biases in Word Embeddings But do not Remove Them](https://arxiv.org/abs/1903.03862)
# - [Word Embeddings, Bias in ML, Why You Don't Like Math, & Why AI Needs You](https://www.youtube.com/watch?v=25nC0n9ERq4&list=PLtmWHNX-gukLQlMvtRJ19s7-8MrnRV6h6&index=9)

# %% [markdown]
# <img src="images/rigler-tweet.png" alt="" style="width: 65%"/>

# %% [markdown]
# ### Fakery

# %% [markdown]
# <img src="images/gpt2-howard.png" alt="" style="width: 65%"/>
#
# [OpenAI's New Multitalented AI writes, translates, and slanders](https://www.theverge.com/2019/2/14/18224704/ai-machine-learning-language-models-read-write-openai-gpt2)

# %% [markdown]
# [He Predicted The 2016 Fake News Crisis. Now He's Worried About An Information Apocalypse.](https://www.buzzfeednews.com/article/charliewarzel/the-terrifying-future-of-fake-news) (interview with Avi Ovadya)
#
# - Generate an audio or video clip of a world leader declaring war. “It doesn’t have to be perfect — just good enough to make the enemy think something happened that it provokes a knee-jerk and reckless response of retaliation.”
#
# - A combination of political botnets and astroturfing, where political movements are manipulated by fake grassroots campaigns to effectively compete with real humans for legislator and regulator attention because it will be too difficult to tell the difference.
#
# - Imagine if every bit of spam you receive looked identical to emails from real people you knew (with appropriate context, tone, etc).

# %% [markdown]
# <img src="images/etzioni-fraud.png" alt="" style="width: 65%"/>
#
# [How Will We Prevent AI-Based Forgery?](https://hbr.org/2019/03/how-will-we-prevent-ai-based-forgery): "We need to promulgate the norm that any item that isn’t signed is potentially forged." 
