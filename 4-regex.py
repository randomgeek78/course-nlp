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
# # Regex

# %% [markdown]
# In this lesson, we'll learn about a useful tool in the NLP toolkit: regex.
#
# Let's consider two motivating examples:
#
# #### 1. The phone number problem
#
# Suppose we are given some data that includes phone numbers:
#
# 123-456-7890
#
# 123 456 7890
#
# 101 Howard
#
# Some of the phone numbers have different formats (hyphens, no hyphens).  Also, there are some errors in the data-- 101 Howard isn't a phon number!  How can we find all the phone numbers?
#
# #### 2. Creating our own tokens
#
# In the previous lessons, we used sklearn or fastai to tokenize our text.  What if we want to do it ourselves?

# %% [markdown]
# ## The phone number problem

# %% [markdown]
# Suppose we are given some data that includes phone numbers:
#
# 123-456-7890
#
# 123 456 7890
#
# (123)456-7890
#
# 101 Howard
#
# Some of the phone numbers have different formats (hyphens, no hyphens, parentheses).  Also, there are some errors in the data-- 101 Howard isn't a phone number!  How can we find all the phone numbers?

# %% [markdown]
# We will attempt this without regex, but will see that this quickly leads to lot of if/else branching statements and isn't a veyr promising approach:

# %% [markdown]
# ### Attempt 1 (without regex)

# %%
phone1 = "123-456-7890"

phone2 = "123 456 7890"

not_phone1 = "101 Howard"

# %%
import string
string.digits


# %%
def check_phone(inp):
    valid_chars = string.digits + ' -()'
    for char in inp:
        if char not in valid_chars:
            return False
    return True


# %%
assert(check_phone(phone1))
assert(check_phone(phone2))
assert(not check_phone(not_phone1))

# %% [markdown]
# ### Attempt 2  (without regex)

# %%
not_phone2 = "1234"

# %%
assert(not check_phone(not_phone2))


# %%
def check_phone(inp):
    nums = string.digits
    valid_chars = nums + ' -()'
    num_counter = 0
    for char in inp:
        if char not in valid_chars:
            return False
        if char in nums:
            num_counter += 1
    if num_counter==10:
        return True
    else:
        return False


# %%
assert(check_phone(phone1))
assert(check_phone(phone2))
assert(not check_phone(not_phone1))
assert(not check_phone(not_phone2))

# %% [markdown]
# ### Attempt 3  (without regex)

# %% [markdown]
# But we also need to extract the digits!

# %% [markdown]
# Also, what about:
#
# 34!NA5098gn#213ee2

# %%
not_phone3 = "34 50 98 21 32"

assert(not check_phone(not_phone3))

# %%
not_phone4 = "(34)(50)()()982132"

assert(not check_phone(not_phone3))

# %% [markdown]
# This is getting increasingly unwieldy.  We need a different approach.

# %% [markdown]
# ## Introducing regex

# %% [markdown]
# Useful regex resources:
#
# - https://regexr.com/
# - http://callumacrae.github.io/regex-tuesday/
# - https://regexone.com/

# %% [markdown]
# **Best practice: Be as specific as possible.**

# %% [markdown]
# Parts of the following section were adapted from Brian Spiering, who taught the MSDS [NLP elective last summer](https://github.com/brianspiering/nlp-course).

# %% [markdown] heading_collapsed=true slideshow={"slide_type": "slide"}
# ### What is regex?
#
# Regular expressions is a pattern matching language. 
#
# Instead of writing `0 1 2 3 4 5 6 7 8 9`, you can write `[0-9]` or `\d`

# %% [markdown] hidden=true slideshow={"slide_type": "fragment"}
# It is Domain Specific Language (DSL). Powerful (but limited language). 
#
# **What other DSLs do you already know?**
# - SQL  
# - Markdown
# - TensorFlow

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Matching Phone Numbers (The "Hello, world!" of Regex)

# %% [markdown] slideshow={"slide_type": "fragment"}
# `[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]` matches US telephone number.

# %% [markdown] slideshow={"slide_type": "slide"}
# Refactored: `\d\d\d-\d\d\d-\d\d\d\d`
#
# A **metacharacter** is one or more special characters that have a unique meaning and are NOT used as literals in the search expression. For example "\d" means any digit.
#
# **Metacharacters are the special sauce of regex.**

# %% [markdown] slideshow={"slide_type": "slide"}
#
#
# Quantifiers
# -----
#
# Allow you to specify how many times the preceding expression should match. 
#
# `{}` is an extact qualifer
#
# Refactored: `\d{3}-\d{3}-\d{4}`

# %% [markdown] slideshow={"slide_type": "slide"}
# Unexact quantifiers
# -----
#
# 1. `?` question mark - zero or one 
# 2. `*` star - zero or more
# 3. `+` plus sign - one or more | 

# %% [markdown]
# ### Regex can look really weird, since it's so concise

# %% [markdown]
# The best (only?) way to learn it is through practice.  Otherwise, you feel like you're just reading lists of rules.
#
# Let's take 15 minutes to begin working through the lessons on [regexone](https://regexone.com/).

# %% [markdown]
# **Reminder: Be as specific as possible!**

# %% [markdown]
# ### Pros & Cons of Regex

# %% [markdown] slideshow={"slide_type": "slide"}
# **What are the advantages of regex?**
#
# 1. Concise and powerful pattern matching DSL
# 2. Supported by many computer languages, including SQL

# %% [markdown] slideshow={"slide_type": "slide"}
# **What are the disadvantages of regex?**
#
# 1. Brittle 
# 2. Hard to write, can get complex to be correct
# 3. Hard to read

# %% [markdown]
# ## Revisiting tokenization

# %% [markdown]
# In the previous lessons, we used a tokenizer.  Now, let's learn how we could do this ourselves, and get a better understanding of tokenization.

# %% [markdown]
# What if we needed to create our own tokens?

# %%
import re

# %%
re_punc = re.compile("([\"\''().,;:/_?!—\-])") # add spaces around punctuation
re_apos = re.compile(r"n ' t ")    # n't
re_bpos = re.compile(r" ' s ")     # 's
re_mult_space = re.compile(r"  *") # replace multiple spaces with just one

def simple_toks(sent):
    sent = re_punc.sub(r" \1 ", sent)
    sent = re_apos.sub(r" n't ", sent)
    sent = re_bpos.sub(r" 's ", sent)
    sent = re_mult_space.sub(' ', sent)
    return sent.lower().split()


# %%
text = "I don't know who Kara's new friend is-- is it 'Mr. Toad'?"

# %%
' '.join(simple_toks(text))

# %%
text2 = re_punc.sub(r" \1 ", text); text2

# %%
text3 = re_apos.sub(r" n't ", text2); text3

# %%
text4 = re_bpos.sub(r" 's ", text3); text4

# %%
re_mult_space.sub(' ', text4)

# %%
sentences = ['All this happened, more or less.',
             'The war parts, anyway, are pretty much true.',
             "One guy I knew really was shot for taking a teapot that wasn't his.",
             'Another guy I knew really did threaten to have his personal enemies killed by hired gunmen after the war.',
             'And so on.',
             "I've changed all their names."]

# %%
tokens = list(map(simple_toks, sentences))

# %%
tokens

# %% [markdown]
# Once we have our tokens, we need to convert them to integer ids.  We will also need to know our vocabulary, and have a way to convert between words and ids.

# %%
import collections

# %%
PAD = 0; SOS = 1

def toks2ids(sentences):
    voc_cnt = collections.Counter(t for sent in sentences for t in sent)
    vocab = sorted(voc_cnt, key=voc_cnt.get, reverse=True)
    vocab.insert(PAD, "<PAD>")
    vocab.insert(SOS, "<SOS>")
    w2id = {w:i for i,w in enumerate(vocab)}
    ids = [[w2id[t] for t in sent] for sent in sentences]
    return ids, vocab, w2id, voc_cnt


# %%
ids, vocab, w2id, voc_cnt = toks2ids(tokens)

# %%
ids

# %%
vocab

# %% [markdown]
# Q: what could be another name of the `vocab` variable above?

# %%
w2id

# %% [markdown] slideshow={"slide_type": "slide"}
# What are the uses of RegEx?
# ---
#
#

# %% [markdown] slideshow={"slide_type": "fragment"}
# 1. Find / Search
# 1. Find & Replace
# 2. Cleaning

# %% [markdown] slideshow={"slide_type": "slide"}
# Don't forgot about Python's `str` methods
# -----
#
# `str.<tab>`
#     
# str.find()

# %% slideshow={"slide_type": "fragment"}
# str.find?

# %% [markdown] slideshow={"slide_type": "slide"}
# Regex vs. String methods
# -----
#
# 1. String methods are easier to understand.
# 1. String methods express the intent more clearly. 
#
# -----
#
# 1. Regex handle much broader use cases.
# 1. Regex can be language independent.
# 1. Regex can be faster at scale.

# %% [markdown]
# ## What about unicode?

# %%
message = "😒🎦 🤢🍕"

re_frown = re.compile(r"😒|🤢")
re_frown.sub(r"😊", message)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Regex Errors:

# %% [markdown] slideshow={"slide_type": "slide"}
# __False positives__ (Type I): Matching strings that we should __not__ have
# matched
#
# __False negatives__ (Type II): __Not__ matching strings that we should have matched

# %% [markdown] slideshow={"slide_type": "slide"}
# Reducing the error rate for a task often involves two antagonistic efforts:
#
# 1. Minimizing false positives
# 2. Minimizing false negatives
#
# **Important to have tests for both!**

# %% [markdown] slideshow={"slide_type": "fragment"}
# In a perfect world, you would be able to minimize both but in reality you often have to trade one for the other.

# %% [markdown] slideshow={"slide_type": "slide"}
# Useful Tools:
# ----
# - [Regex cheatsheet](http://www.cheatography.com/davechild/cheat-sheets/regular-expressions/)
# - [regexr.com](http://regexr.com/) Realtime regex engine
# - [pyregex.com](https://pythex.org/) Realtime Python regex engine

# %% [markdown] slideshow={"slide_type": "slide"}
# Summary
# ----
#
# 1. We use regex as a metalanguage to find string patterns in blocks of text
# 1. `r""` are your IRL friends for Python regex
# 1. We are just doing binary classification so use the same performance metrics
# 1. You'll make a lot of mistakes in regex 😩. 
#     - False Positive: Thinking you are right but you are wrong
#     - False Negative: Missing something

# %% [markdown] slideshow={"slide_type": "slide"}
# <center><img src="images/face_tat.png" width="700"/></center>

# %% [markdown] slideshow={"slide_type": "slide"}
# <br>
# <br>
# ---

# %% [markdown]
# <center><img src="https://imgs.xkcd.com/comics/perl_problems.png" width="700"/></center>

# %% [markdown]
# <center><img src="https://imgs.xkcd.com/comics/regex_golf.png" width="700"/></center>

# %% [markdown] slideshow={"slide_type": "slide"}
# Regex Terms
# ----
#

# %% [markdown] slideshow={"slide_type": "fragment"}
# - __target string__:	This term describes the string that we will be searching, that is, the string in which we want to find our match or search pattern.
#

# %% [markdown] slideshow={"slide_type": "fragment"}
# - __search expression__: The pattern we use to find what we want. Most commonly called the regular expression. 
#

# %% [markdown] slideshow={"slide_type": "slide"}
# - __literal__:	A literal is any character we use in a search or matching expression, for example, to find 'ind' in 'windows' the 'ind' is a literal string - each character plays a part in the search, it is literally the string we want to find.

# %% [markdown] slideshow={"slide_type": "fragment"}
# - __metacharacter__: A metacharacter is one or more special characters that have a unique meaning and are NOT used as literals in the search expression. For example "." means any character.
#
# Metacharacters are the special sauce of regex.

# %% [markdown] slideshow={"slide_type": "slide"}
# - __escape sequence__:	An escape sequence is a way of indicating that we want to use a metacharacters as a literal. 

# %% [markdown] slideshow={"slide_type": "fragment"}
# In a regular expression an escape sequence involves placing the metacharacter \ (backslash) in front of the metacharacter that we want to use as a literal. 
#
# `'\.'` means find literal period character (not match any character)

# %% [markdown]
# Regex Workflow
# ---
# 1. Create pattern in Plain English
# 2. Map to regex language
# 3. Make sure results are correct:
#     - All Positives: Captures all examples of pattern
#     - No Negatives: Everything captured is from the pattern
# 4. Don't over-engineer your regex. 
#     - Your goal is to Get Stuff Done, not write the best regex in the world
#     - Filtering before and after are okay.
