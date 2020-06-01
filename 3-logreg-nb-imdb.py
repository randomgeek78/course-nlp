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
# # Sentiment Classification the old-fashioned way: 
# ## `Naive Bayes`, `Logistic Regression`, and `Ngrams`

# %% [markdown]
# The purpose of this notebook is to show how sentiment classification is done via the classic techniques of `Naive Bayes`, `Logistic regression`, and `Ngrams`.  We will be using `sklearn` and the `fastai` library.
#
# In a future lesson, we will revisit sentiment classification using `deep learning`, so that you can compare the two approaches.

# %% [markdown]
# The content here was extended from [Lesson 10 of the fast.ai Machine Learning course](https://course.fast.ai/lessonsml1/lesson10.html). Linear model is pretty close to the state of the art here.  Jeremy surpassed state of the art using a RNN in fall 2017.

# %% [markdown] heading_collapsed=true
# ## 0.The fastai library

# %% [markdown] hidden=true
# We will begin using [the fastai library](https://docs.fast.ai) (version 1.0) in this notebook.  We will use it more once we move on to neural networks.
#
# The fastai library is built on top of PyTorch and encodes many state-of-the-art best practices. It is used in production at a number of companies.  You can read more about it here:
#
# - [Fast.ai's software could radically democratize AI](https://www.zdnet.com/article/fast-ais-new-software-could-radically-democratize-ai/) (ZDNet)
#
# - [fastai v1 for PyTorch: Fast and accurate neural nets using modern best practices](https://www.fast.ai/2018/10/02/fastai-ai/) (fast.ai)
#
# - [fastai docs](https://docs.fast.ai/)
#
# ### Installation
#
# With conda:
#
# `conda install -c pytorch -c fastai fastai=1.0`
#
# Or with pip:
#
# `pip install fastai==1.0`
#
# More [installation information here](https://github.com/fastai/fastai/blob/master/README.md).
#
# Beginning in lesson 4, we will be using GPUs, so if you want, you could switch to a [cloud option](https://course.fast.ai/#using-a-gpu) now to setup fastai.

# %% [markdown]
# ## 1. The IMDB dataset

# %% [markdown]
# <img src="IMDb.png" alt="floating point" style="width: 90%"/>

# %% [markdown]
# The [large movie review dataset](http://ai.stanford.edu/~amaas/data/sentiment/) contains a collection of 50,000 reviews from IMDB, We will use the version hosted as part [fast.ai datasets](https://course.fast.ai/datasets.html) on AWS Open Datasets. 
#
# The dataset contains an even number of positive and negative reviews. The authors considered only highly polarized reviews. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. Neutral reviews are not included in the dataset. The dataset is divided into training and test sets. The training set is the same 25,000 labeled reviews.
#
# The **sentiment classification task** consists of predicting the polarity (positive or negative) of a given text.

# %% [markdown]
# ### Imports

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# %%
from fastai import *
from fastai.text import *
from fastai.utils.mem import GPUMemTrace #call with mtrace

# %%
import sklearn.feature_extraction.text as sklearn_text
import pickle 

# %% [markdown]
# ### Preview the sample IMDb data set

# %% [markdown]
# fast.ai has a number of [datasets hosted via AWS Open Datasets](https://course.fast.ai/datasets.html) for easy download. We can see them by checking the docs for URLs (remember `??` is a helpful command):

# %%
?? URLs

# %% [markdown]
# It is always good to start working on a sample of your data before you use the full dataset-- this allows for quicker computations as you debug and get your code working. For IMDB, there is a sample dataset already available:

# %%
path = untar_data(URLs.IMDB_SAMPLE)
path

# %% [markdown]
# #### Read the data set into a pandas dataframe, which we can inspect to get a sense of what our data looks like. We see that the three columns contain review label, review text, and the `is_valid` flag, respectively. `is_valid` is a boolean flag indicating whether the row is from the validation set or not.

# %%
df = pd.read_csv(path/'texts.csv')
df.head()

# %% [markdown]
# ### Extract the movie reviews from the sample IMDb data set.
# #### We will be using [TextList](https://docs.fast.ai/text.data.html#TextList) from the fastai library:

# %%
# %%time
# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!

count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        movie_reviews = (TextList.from_csv(path, 'texts.csv', cols='text')
                         .split_from_df(col=2)
                         .label_from_df(cols=0))
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')


# %% [markdown]
# ### Exploring IMDb review data

# %% [markdown]
# A good first step for any data problem is to explore the data and get a sense of what it looks like.  In this case we are looking at movie reviews, which have been labeled as "positive" or "negative". The reviews have already been `tokenized`, i.e. split into `tokens`, basic units such as words, prefixes, punctuation, capitalization, and other features of the text.

# %% [markdown]
# ### Let's examine the`movie_reviews` object:

# %%
dir(movie_reviews)

# %% [markdown]
# ### `movie_reviews` splits the data into training and validation sets, `.train` and `.valid` 

# %%
print(f'There are {len(movie_reviews.train.x)} and {len(movie_reviews.valid.x)} reviews in the training and validations sets, respectively.')

# %% [markdown]
# ### Reviews are composed of lists of tokens. In NLP, a **token** is the basic unit of processing (what the tokens are depends on the application and your choices). Here, the tokens mostly correspond to words or punctuation, as well as several special tokens, corresponding to unknown words, capitalization, etc.

# %% [markdown]
# ### Special tokens:
# All those tokens starting with "xx" are fastai special tokens.  You can see the list of all of them and their meanings ([in the fastai docs](https://docs.fast.ai/text.transform.html)): 
#
# ![image.png](attachment:image.png)

# %% [markdown]
# ### Let's examine the structure of the `training set`

# %% [markdown]
# #### movie_reviews.train is a `LabelList` object. 
# #### movie_reviews.train.x  is a `TextList` object that holds the reviews
# #### movie_reviews.train.y is a `CategoryList` object that holds the labels 

# %%
print(f'\fThere are {len(movie_reviews.train.x)} movie reviews in the training set\n')
print(movie_reviews.train)

# %% [markdown]
# #### The text of the movie review is stored as a character `string`, which contains the tokens separated by spaces. Here is the text of the first review:

# %%
print(movie_reviews.train.x[0].text)
print(f'\nThere are {len(movie_reviews.train.x[0].text)} characters in the review')

# %% [markdown]
# #### The text string can be split to get the list of tokens.

# %%
print(movie_reviews.train.x[0].text.split())
print(f'\nThe review has {len(movie_reviews.train.x[0].text.split())} tokens')

# %% [markdown]
# #### The review tokens are `numericalized`, ie. mapped to integers.  So a movie review is also stored as an array of integers:

# %%
print(movie_reviews.train.x[0].data)
print(f'\nThe array contains {len(movie_reviews.train.x[0].data)} numericalized tokens')

# %% [markdown]
# ## 2. The IMDb Vocabulary

# %% [markdown]
# ### The `movie_revews` object also contains a `.vocab` property, even though it is not shown with`dir()`. (This may be an error in the `fastai` library.) 

# %%
movie_reviews.vocab

# %% [markdown]
# ### The `vocab` object is a kind of reversible dictionary that translates back and forth between tokens and their integer representations.  It has two methods of particular interest: `stoi` and `itos`, which stand for `string-to-index` and `index-to-string`

# %% [markdown]
# #### `movie_reviews.vocab.stoi` maps vocabulary tokens to their `indexes` in vocab

# %%
movie_reviews.vocab.stoi

# %% [markdown]
# #### `movie_reviews.vocab.itos` maps the `indexes` of vocabulary tokens to `strings`

# %%
movie_reviews.vocab.itos

# %% [markdown]
# #### Notice that ints-to-string and string-to-ints have different lengths.  Think for a moment about why this is.
# See Hint below

# %%
print('itos ', 'length ',len(movie_reviews.vocab.itos),type(movie_reviews.vocab.itos) )
print('stoi ', 'length ',len(movie_reviews.vocab.stoi),type(movie_reviews.vocab.stoi) )

# %% [markdown]
# #### Hint: `stoi` is an instance of the class `defaultdict`
# <img src="default_dict.png" alt="floating point" style="width: 90%"/>

# %% [markdown]
# #### In a `defaultdict`, rare words that appear fewer than three times in the corpus, and words that are not in the dictionary, are mapped to a `default value`, in this case, zero

# %%
rare_words = ['acrid','a_random_made_up_nonexistant_word','acrimonious','allosteric','anodyne','antikythera']
for word in rare_words:
    print(movie_reviews.vocab.stoi[word])

# %% [markdown]
# #### What's the `token` corresponding to the `default` value?

# %%
print(movie_reviews.vocab.itos[0])

# %% [markdown]
# #### Note that `stoi` (string-to-int) is larger than `itos` (int-to-string).

# %%
print(f'len(stoi) = {len(movie_reviews.vocab.stoi)}')
print(f'len(itos) = {len(movie_reviews.vocab.itos)}')
print(f'len(stoi) - len(itos) = {len(movie_reviews.vocab.stoi) - len(movie_reviews.vocab.itos)}')

# %% [markdown]
# #### This is because many words map to `unknown`.  We can confirm here:

# %%
unk = []
for word, num in movie_reviews.vocab.stoi.items():
    if num==0:
        unk.append(word)

# %%
len(unk)

# %% [markdown]
# #### Question: why isn't len(unk) = len(stoi) - len(itos)?
# Hint: remember the list of rare words we used to query `stoi` a few cells back?

# %% [markdown]
# #### Here are the first 25 words that are mapped to `unknown`

# %%
unk[:25]

# %% [markdown]
# ## 3. Map the movie reviews into a vector space

# %% [markdown]
# ### There are 6016 unique tokens in the IMDb review vocabulary. Their numericalized values range from 0 to 6015

# %%
print(f'There are {len(movie_reviews.vocab.itos)} unique tokens in the IMDb review sample vocabulary')
print(f'The numericalized token values run from {min(movie_reviews.vocab.stoi.values())} to {max(movie_reviews.vocab.stoi.values())} ')

# %% [markdown]
# ### Each review can be mapped to a 6016-dimensional `embedding vector` whose indices correspond to the numericalized tokens, and whose values are the number of times the corresponding token appeared in the review. To do this efficiently we need to learn a bit about `Counters`.

# %% [markdown]
# ### 3A. Counters

# %% [markdown]
# A **Counter** is a useful Python object.  A **Counter** applied to a list returns an ordered dictionary whose keys are the unique elements in the list, and whose values are the counts of the unique elements. Counters are from the collections module (along with OrderedDict, defaultdict, deque, and namedtuple).
# Here is how Counters work:

# %% [markdown]
# #### Let's make a TokenCounter for movie reviews

# %%
TokenCounter = lambda review_index : Counter((movie_reviews.train.x)[review_index].data)
TokenCounter(0).items()

# %% [markdown]
# #### The TokenCounter `keys` are the numericalized `tokens` that apper in the review

# %%
TokenCounter(0).keys()

# %% [markdown]
# #### The TokenCounter `values` are the `token multiplicities`, i.e the number of times each `token` appears in the review

# %%
TokenCounter(0).values()

# %% [markdown]
# ### 3B. Mapping movie reviews to `embedding vectors`

# %% [markdown]
# #### Make a `count_vectorizer` function that represents a movie review as a 6016-dimensional `embedding vector`
# #### The `indices` of  the `embedding vector` correspond to the n6016 numericalized tokens in the vocabulary; the `values` specify how often the corresponding token appears in the review. 

# %%
n_terms = len(movie_reviews.vocab.itos)
n_docs = len(movie_reviews.train.x)
make_token_counter = lambda review_index: Counter(movie_reviews.train.x[review_index].data)
def count_vectorizer(review_index,n_terms = n_terms,make_token_counter = make_token_counter):
    # input: review index, n_terms, and tokenizer function
    # output: embedding vector for the review
    embedding_vector = np.zeros(n_terms)        
    keys = list(make_token_counter(review_index).keys())
    values = list(make_token_counter(review_index).values())
    embedding_vector[keys] = values
    return embedding_vector

# make the embedding vector for the first review
embedding_vector = count_vectorizer(0)

# %% [markdown]
# #### Here is the `embedding vector` for the first review in the training data set

# %%
print(f'The review is embedded in a {len(embedding_vector)} dimensional vector')
embedding_vector

# %% [markdown]
# ## 4. Create the document-term matrix for the IMDb

# %% [markdown]
# #### In non-deep learning methods of NLP, we are often interested only in `which words` were used in a review, and `how often each word got used`. This is known as the `bag of words` approach, and it suggests a really simple way to store a document (in this case, a movie review). 
#
# #### For each review we can keep track of which words were used and how often each word was used with a `vector` whose `length` is the number of tokens in the vocabulary, which we will call `n`. The `indexes` of this `vector` correspond to the `tokens` in the `IMDb vocabulary`, and the`values` of the vector are the number of times the corresponding tokens appeared in the review. For example the values stored at indexes 0, 1, 2, 3, 4 of the vector record the number of times the 5 tokens ['xxunk','xxpad','xxbos','xxeos','xxfld'] appeared in the review, respectively.
#
# #### Now, if our movie review database has `m` reviews, and each review is represented by a `vector` of length `n`, then vertically stacking the row vectors for all the reviews creates a matrix representation of the IMDb, which we call its `document-term matrix`. The `rows` correspond to `documents` (reviews), while the `columns` correspond to `terms` (or tokens in the vocabulary).

# %% [markdown]
# In the previous lesson, we used [sklearn's CountVectorizer](https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/feature_extraction/text.py#L940) to generate the `vectors` that represent individual reviews. Today we will create our own (similar) version.  This is for two reasons:
# - to understand what sklearn is doing underneath the hood
# - to create something that will work with a fastai TextList

# %% [markdown]
# ### Form the embedding vectors for the movie_reviews in the training set and stack them vertically

# %%
# Define a function to build the full document-term matrix
print(f'there are {n_docs} reviews, and {n_terms} unique tokens in the vocabulary')
def make_full_doc_term_matrix(count_vectorizer,n_terms=n_terms,n_docs=n_docs):
    
    # loop through the movie reviews
    for doc_index in range(n_docs):
        
        # make the embedding vector for the current review
        embedding_vector = count_vectorizer(doc_index,n_terms)    
            
        # append the embedding vector to the document-term matrix
        if(doc_index == 0):
            A = embedding_vector
        else:
            A = np.vstack((A,embedding_vector))
            
    # return the document-term matrix
    return A

# Build the full document term matrix for the movie_reviews training set
A = make_full_doc_term_matrix(count_vectorizer)

# %% [markdown]
# ### Explore the `sparsity` of the document-term matrix

# %% [markdown]
# #### The `sparsity` of a matrix is defined as the fraction of of zero-valued elements

# %%
NNZ = np.count_nonzero(A)
sparsity = (A.size-NNZ)/A.size
print(f'Only {NNZ} of the {A.size} elements in the document-term matrix are nonzero')
print(f'The sparsity of the document-term matrix is {sparsity}')

# %% [markdown]
# #### Using matplotlib's `spy` method, we can visualize the structure of the `document-term matrix`
# `spy` plots the array, indicating each non-zero value with a dot.

# %%
fig = plt.figure()
plt.spy(A, markersize=0.10, aspect = 'auto')
fig.set_size_inches(8,6)
fig.savefig('doc_term_matrix.png', dpi=800)


# %% [markdown]
# #### Several observations stand out:
# 1. Evidently, the document-term matrix is `sparse` ie. has a high proportion of zeros! 
# 2. The density of the matrix increases toward the `left` edge. This makes sense because the tokens are ordered by usage frequency, with frequency increasing toward the `left`.
# 3. There is a perplexing pattern of curved vertical `density ripples`. If anyone has an explanation, please let me know! 
#
# #### Next we'll see how to  exploit matrix sparsity to save memory storage space, and compute time and resources.
#

# %% [markdown]
# ## 5. Sparse Matrix Representation

# %% [markdown]
# #### Even though we've reduced over 19,000 unique words in our corpus of reviews down to a vocabulary of 6,000 words, that's still a lot! But reviews are generally short, a few hundred words. So most tokens don't appear in a typical review.  That means that most of the entries in the document-term matrix will be zeros, and therefore ordinary matrix operations will waste a lot of compute resources multiplying and adding zeros. 
#
# ####  We want to maximize the use of space and time by storing and performing matrix operations on our document-term matrix as a **sparse matrix**. `scipy` provides tools for efficient sparse matrix representatin and operations. 

# %% [markdown]
# #### Loosely speaking,  matrix with a high proportion of zeros is called `sparse` (the opposite of sparse is `dense`).  For sparse matrices, you can save a lot of memory by only storing the non-zero values.
#
# #### More specifically, a class of matrices is called **sparse** if the number of non-zero elements is proportional to the number of rows (or columns) instead of being proportional to the product rows x columns. An example is the class of diagonal matrices.
#
#
# <img src="images/sparse.png" alt="floating point" style="width: 30%"/>
#
#

# %% [markdown]
# ### Visualizing sparse matrix structure
# <img src="sparse-matrix-structure-visualization.png" alt="floating point" style="width: 90%"/>
# ref. https://scipy-lectures.org/advanced/scipy_sparse/introduction.html

# %% [markdown]
# ### Sparse matrix storage formats
#
# <img src="summary_of_sparse_matrix_storage_schemes.png" alt="floating point" style="width: 90%"/>
# ref. https://scipy-lectures.org/advanced/scipy_sparse/storage_schemes.html
#
# There are the most common sparse storage formats:
# - coordinate-wise (scipy calls COO)
# - compressed sparse row (CSR)
# - compressed sparse column (CSC)
#
#

# %% [markdown]
# ### Definition of the Compressed Sparse Row (CSR) format
#
# Let's start out with a presecription for the **CSR format** (ref. https://en.wikipedia.org/wiki/Sparse_matrix)
#
# Given a full matrix **`A`** that has **`m`** rows, **`n`** columns, and **`N`** nonzero values, the CSR (Compressed Sparse Row) representation uses three arrays as follows:
#
# 1. **`Val[0:N]`** contains the **values** of the **`N` non-zero elements**.
#
# 2. **`Col[0:N]`** contains the **column indices** of the **`N` non-zero elements**. 
#     
# 3. For each row **`i`** of **`A`**, **`RowPointer[i]`** contains the index in **Val** of the the first **nonzero value** in row **`i`**. If there are no nonzero values in the **ith** row, then **`RowPointer[i] = None`**. And, by convention, an extra value **`RowPointer[m] = N`** is tacked on at the end. 
#
# Question: How many floats and ints does it take to store the matrix **`A`** in CSR format?
#
# Let's walk through [a few examples](http://www.mathcs.emory.edu/~cheung/Courses/561/Syllabus/3-C/sparse.html) at the Emory University website
#
#

# %% [markdown]
# ## 6. Store the document-term matrix in CSR format
# i.e. given the `TextList` object containing the list of reviews, return the three arrays (values, column_indices, row_pointer)

# %% [markdown]
# ### Scipy Implementation of sparse matrices
#
# From the [Scipy Sparse Matrix Documentation](https://docs.scipy.org/doc/scipy-0.18.1/reference/sparse.html)
#
# - To construct a matrix efficiently, use either dok_matrix or lil_matrix. The lil_matrix class supports basic slicing and fancy indexing with a similar syntax to NumPy arrays. As illustrated below, the COO format may also be used to efficiently construct matrices
# - To perform manipulations such as multiplication or inversion, first convert the matrix to either CSC or CSR format.
# - All conversions among the CSR, CSC, and COO formats are efficient, linear-time operations.

# %% [markdown]
# ### To really understand the CSR format, we need to be able know how to do two things:
# 1. Translate a regular matrix A into CSR format
# 2. Reconstruct a regular matrix from its CSR sparse representation
#

# %% [markdown]
# ### 6.1. Translate a regular matrix A into CSR format
# This is done by implementing the definition of `CSR format`, given above.

# %%
# construct the document-term matrix in CSR format
# i.e. return (values, column_indices, row_pointer)
def get_doc_term_matrix(text_list, n_terms):
    
    # inputs:
    #    text_list, a TextList object
    #    n_terms, the number of tokens in our IMDb vocabulary
    
    # output: 
    #    the CSR format sparse representation of the document-term matrix in the form of a
    #    scipy.sparse.csr.csr_matrix object

    
    # initialize arrays
    values = []
    column_indices = []
    row_pointer = []
    row_pointer.append(0)

    # from the TextList object
    for _, doc in enumerate(text_list):
        feature_counter = Counter(doc.data)
        column_indices.extend(feature_counter.keys())
        values.extend(feature_counter.values())
        # Tack on N (number of nonzero elements in the matrix) to the end of the row_pointer array
        row_pointer.append(len(values))
        
    return scipy.sparse.csr_matrix((values, column_indices, row_pointer),
                                   shape=(len(row_pointer) - 1, n_terms),
                                   dtype=int)

# %% [markdown]
# #### Get the document-term matrix in CSR format for the training data

# %%
# %%time
train_doc_term = get_doc_term_matrix(movie_reviews.train.x, len(movie_reviews.vocab.itos))

# %%
type(train_doc_term)

# %%
train_doc_term.shape

# %% [markdown]
# #### Get the document-term matrix in CSR format for the validation data

# %%
# %%time
valid_doc_term = get_doc_term_matrix(movie_reviews.valid.x, len(movie_reviews.vocab.itos))

# %%
type(valid_doc_term)

# %%
valid_doc_term.shape


# %% [markdown]
# ### 6.2 Reconstruct a regular matrix from its CSR sparse representation
# #### Given a CSR format sparse matrix representation $(\text{values},\text{column_indices}, \text{row_pointer})$ of a $\text{m}\times \text{n}$ matrix $\text{A}$, <br> how can we recover $\text{A}$?
#
# First create $\text{m}\times \text{n}$ matrix with all zeros.
# We will recover $\text{A}$ by overwriting the entries in the zeros matrix row by row with the non-zero entries in $\text{A}$ as follows:

# %%
def CSR_to_full(values, column_indices, row_ptr, m,n):
    A = zeros(m,n)
    for row in range(n):
        if row_ptr is not null:
            A[row,column_indices[row_ptr[row]:row_ptr[row+1]]] = values[row_ptr[row]:row_ptr[row+1]]
    return A



# %% [markdown]
# ## 7. IMDb data exploration exercises

# %% [markdown]
# #### The`.todense()` method converts a sparse matrix back to a regular (dense) matrix.

# %%
valid_doc_term

# %%
valid_doc_term.todense()[:10,:10]

# %% [markdown]
# #### Consider the second review in the validation set

# %%
review = movie_reviews.valid.x[1]
review

# %% [markdown]
# **Exercise 1:** How many times does the word "it" appear in this review? Confirm that the correct values is stored in the document-term matrix, for the row corresponding to this review and the column corresponding to the word "it".

# %% [markdown]
# #### Answer 1:

# %%
# try it! 
# Your code here.

# %% [markdown]
# **Exercise 2**: Confirm that the review has 144 tokens, 81 of which are distinct

# %% [markdown]
# #### Answer 2:

# %%
valid_doc_term[1]

# %%
valid_doc_term[1].sum()

# %%
len(set(review.data))

# %% [markdown]
# **Exercise 3:** How could you convert review.data back to text (without just using review.text)?

# %%
review.data

# %% [markdown]
# #### Answer 3:

# %%
word_list = [movie_reviews.vocab.itos[a] for a in review.data]
print(word_list)

# %%
reconstructed_text = ' '.join(word_list)
print(reconstructed_text)

# %% [markdown]
# ## *Video 4 material ends here.* 
# ## *Video 5 material begins below.*

# %% [markdown]
# ## 8. What is a [Naive Bayes classifier](https://towardsdatascience.com/the-naive-bayes-classifier-e92ea9f47523)? 

# %% [markdown]
#
# #### The `bag of words model` considers a movie review as equivalent to a list of the counts of all the tokens that it contains. When you do this, you throw away the rich information that comes from the sequential arrangement of the tokens into sentences and paragraphs. 
#
# #### Nevertheless, even if you are not allowed to read the review but are only given its representation as `token counts`, you can usually still get a pretty good sense of whether the review was good or bad. How do you do this?  By mentally gauging the overall `positive` or `negative` sentiment that the collection of words conveys, right?  
#
# #### The `Naive Bayes Classifier` is an algorithm that encodes this simple reasoning process mathematically. It is based on two important pieces of information that we can learn from the training set:
# * The `class priors`, i.e. the probabilities that a randomly chosen review will be `positive`, or `negative`
# * The `token likelihoods` i.e. how likely is it that a given token would appear in a `positive` or `negative` review 
#
# #### It turns out that this is all the information we need to build a model capable of predicting fairly accurately how any given review will be classified, given its text! 
#
# #### We shall unfold the complete explanation of the magic of the Naive Bayes Classifier in the next section. 
#
# #### Meanwhile, In this section, we focus on how to compute the necessary information from the training data, specifically the `prior probabilities` for reviews of each class, and the `class occurrence counts` and `class likelihood ratios` for each `token` in the `vocabulary`. 

# %% [markdown]
# ### 8A. Class priors

# %% [markdown]
# #### From the training data we can determine the `class priors` $p$ and $q$, which are the overall probabilities that a randomly chosen review is in the `positive`, or `negative` class, resepectively. 
#
# #### $p=\frac{N^{+}}{N}$ 
# #### and
# #### $q=\frac{N^{-}}{N}$ 
#
# #### Here $N^{+}$ and $N^{-}$ are the numbers of `positive` and `negative` reviews, and $N$ is the total number of reviews in the training set, so that 
#
# #### $N = N^{+} + N^{-}$, 
#
# #### and 
#
# #### $q = 1-p$

# %% [markdown]
# ### 8B. Class `occurrence counts`

# %% [markdown]
# #### Let $C^{+}_{t}$ and $C^{-}_{t}$ be the `occurrence counts` of token $t$ in `positive` and `negative` reviews, respectively, and $N^{+}$ and $N^{-}$ be the total numbers of`positive` and `negative` reviews in the data set, respectively. 
#

# %% [markdown]
# ### 8B.1 Data exploration with class `occurrence counts`

# %% [markdown]
# #### Movie reviews classes and their integer representations

# %%
dir(movie_reviews)

# %%
movie_reviews.y.c

# %%
movie_reviews.y.classes

# %%
positive = movie_reviews.y.c2i['positive']
negative = movie_reviews.y.c2i['negative']
print(f'Integer representations:  positive: {positive}, negative: {negative}')

# %% [markdown]
# #### Brief names for training set document term matrix and its labels, validation labels, and vocabulary

# %%
x = train_doc_term
y = movie_reviews.train.y
valid_y = movie_reviews.valid.y
v = movie_reviews.vocab

# %%
x.shape

# %% [markdown]
# #### The `count arrays` `C1` and `C0` list the total `occurrence counts` of the tokens in `positive` and `negative` reviews, respectively.

# %%
C1 = np.squeeze(np.asarray(x[y.items==positive].sum(0)))
C0 = np.squeeze(np.asarray(x[y.items==negative].sum(0)))

# %% [markdown]
# For each vocabulary token, we are summing up how many positive reviews it is in, and how many negative reviews it is in. Here are the occurrence counts for the first 10 tokens in the vocabulary.

# %%
print(C1[:10])
print(C0[:10])

# %% [markdown]
# ### 8B.2 Exercise

# %% [markdown]
# #### We can use `C0` and `C1` to do some more data exploration!

# %% [markdown]
# **Exercise 4**: Compare how often the word "loved" appears in positive reviews vs. negative reviews.  Do the same for the word "hate"

# %% [markdown]
# #### Answer 4:

# %%
# Exercise: How often does the word "love" appear in neg vs. pos reviews?
ind = v.stoi['love']
pos_counts = C1[ind] 
neg_counts = C0[ind] 
print(f'The word "love" appears {pos_counts} and {neg_counts} times in positive and negative documents, respectively')

# %%
# Exercise: How often does the word "hate" appear in neg vs. pos reviews?
ind = v.stoi['hate']
pos_counts = C1[ind] 
neg_counts = C0[ind] 
print(f'The word "hate" appears {pos_counts} and {neg_counts} times in positive and negative documents, respectively')

# %% [markdown]
# #### Let's look for an example of a positive review containing the word "hated"

# %%
index = v.stoi['hated']
a = np.argwhere((x[:,index] > 0))[:,0]
print(a)
b = np.argwhere(y.items==positive)[:,0]
print(b)
c = list(set(a).intersection(set(b)))[0]
review = movie_reviews.train.x[c]
review.text

# %% [markdown]
# #### Example of a negative review with the word "loved"

# %%
index = v.stoi['loved']
a = np.argwhere((x[:,index] > 0))[:,0]
print(a)
b = np.argwhere(y.items==negative)[:,0]
print(b)
c = list(set(a).intersection(set(b)))[0]
review = movie_reviews.train.x[c]
review.text

# %% [markdown]
# ### 8C. Class likelihood ratios

# %% [markdown]
# #### Then, given the knowledge that a review is classified as `positive`, the `conditional likelihood` that a token $t$ will appear in the review is
# ### $ L(t|+) = \frac{C^{+}_{t}}{N^+}$, 
# #### and simlarly, the `conditional likelihood` of a token appearing in a `negative` review is 
# ### $ L(t|-) = \frac{C^{-}_{t}}{N^-}$

# %% [markdown]
# ### 8D. The `log-count ratio`

# %% [markdown]
# #### From the class likelihood ratios, we can define a **log-count ratio** $R_{t}$ for each token $t$ as
# ### $ R_{t} = \text{log} \frac{L(t|+)}  {L(t|-)}$
# #### The `log-count ratio` ranks tokens by their relative affinities for positive and negative reviews
# #### We observe that
# * $R_{t} \gt 0$ means `positive` reviews are more likely to contain this token 
# * $R_{t} \lt 0$ means `negative` reviews are more likely to contain this token 
# * $R_{t} = 0$ indicates the token $t$ has equal likelihood to appear in  `positive` and `negative` reviews
#

# %% [markdown]
# ## 9. Building a Naive Bayes Classifier for IMDb movie reviews

# %% [markdown]
# #### From the `occurrence count` arrays, we can compute the `class likelihoods` and `log-count ratios` of all the tokens in the vocabulary. 

# %% [markdown]
# ### 9A. Compute the `class likelihoods`

# %% [markdown]
# #### We compute slightly modified `conditional likelihoods`, by adding 1 to the numerator and denominator to insure numerically stability.

# %%
L1 = (C1+1) / ((y.items==positive).sum() + 1)
L0 = (C0+1) / ((y.items==negative).sum() + 1)

# %% [markdown]
# ### 9B. Compute the `log-count ratios`

# %% [markdown]
# #### The log-count ratios are

# %%
R = np.log(L1/L0)
print(R)

# %% [markdown]
# #### Data Exercise: find the vocabulary words most likely to be associated with positive and negative reviews

# %% [markdown]
# #### Get the indices of the tokens with the highest and lowest log-count ratios

# %%
n_tokens = 10
highest_R = np.argpartition(R, -n_tokens)[-n_tokens:]
lowest_R = np.argpartition(R, n_tokens)[:n_tokens]

# %%
print(f'Highest {n_tokens} log-count ratios: {R[list(highest_R)]}\n')
print(f'Lowest {n_tokens} log-count ratios: {R[list(lowest_R)]}')

# %% [markdown]
# #### Most positive words:

# %%
highest_R

# %%
[v.itos[k] for k in highest_R]

# %% [markdown]
# #### There are only two movie reviews that mention "biko"

# %%
token = 'biko'
train_doc_term[:,v.stoi[token]]

# %% [markdown]
# #### Which movie review has the most occurrences of 'biko'?

# %%
index = np.argmax(train_doc_term[:,v.stoi[token]])
n_times = train_doc_term[index,v.stoi[token]]
print(f'review # {index} has {n_times} occurrences of "{token}"\n')
print(movie_reviews.train.x[index].text)

# %% [markdown]
# #### Most negative words:

# %%
lowest_R

# %%
[v.itos[k] for k in lowest_R]

# %% [markdown]
# #### There's only one movie review that mentions "soderbergh"

# %%
token = 'soderbergh'
train_doc_term[:,v.stoi[token]]

# %%
index = np.argmax(train_doc_term[:,v.stoi[token]])
n_times = train_doc_term[index,v.stoi[token]]
print(f'review # {index} has {n_times} occurrences of "{token}"\n')
print(movie_reviews.train.x[index].text)


# %%
train_doc_term[:,v.stoi[token]]

# %% [markdown]
# ### 9C. Compute the prior probabilities for  each class

# %%
p = (y.items==positive).mean()
q = (y.items==negative).mean()
print(f'The prior probabilities for positive and negative classes are {p} annd {q}')

# %% [markdown]
# #### The log probability ratio is
#
# ### $b = \text{log} \frac{p} {q}$ 
#
# #### is a measure of the `bias`, or `imbalance` in the data set. 
#
# * $b = 0$ indicates a perfectly balanced data set
# * $b \gt 0$ indicates bias towards `positive` reviews 
# * $b \lt 0$ indicates bias towards `negative` reviews 

# %%
b = np.log((y.items==positive).mean() / (y.items==negative).mean())
print(f'The log probability ratio is L = {b}')

# %% [markdown]
# #### We see that the training set is slightly imbalanced toward `negative` reviews.

# %% [markdown]
# ### 9D.  Putting it all together: the Naive Bayes Movie Review Classifier
# In this section, we'll start with a discussion of Bayes' Theorem, then we'll use it to derive the Naive Bayes Classifier. Next we'll apply the Naive Bayes classifier to our movie reviews problem. Finally we'll review the prescription for building a Naive Bayes Classifier. 

# %% [markdown]
# ### 9D.1 What is Bayes Theorem, and what does it have to say about IMDb movie reviews?
#
# Consider two events, $A$ and $B$  
# Then the probability of $A$ and $B$ occurring together can be written in two ways:
# $p(A,B) = p(A|B)\cdot p(B)$
# $p(A,B) = p(B|A)\cdot p(A)$
#
# where $p(A|B)$ and $p(B|A)$ are conditional probabilities:
# $p(A|B)$ is the probability of $A$ occurring given that $B$ has occurred,
# $p(A)$ is the probability that $A$ occurs,
# $p(B)$ is the probabilityt that $B$ occurs
#
#
# $\textbf{Bayes Theorem}$ is just the statement that the right hand sides of the above two equations are equal:
#
# $p(A|B) \cdot p(B) = p(B|A) \cdot p(A)$
#
# Applying $\textbf{Bayes Theorem}$ to our IMDb movie review problem:
#
# We identify $A$ and $B$ as <br> 
# $A \equiv \text{class}$, i.e. positive or negative, and <br>
# $B \equiv \text{tokens}$, i.e. the "bag" of tokens used in the review
#
# Then $\textbf{Bayes Theorem}$ says
#
# $p(\text{class}|\text{tokens})\cdot p(\text{tokens}) = p(\text{tokens}|\text{class}) \cdot p(\text{class})$
#
# so that <br>
# $p(\text{class}|\text{tokens}) = p(\text{tokens}|\text{class})\cdot \frac{p(\text{class})}{p(\text{tokens})}$
#
# Since $p(\text{tokens})$ is a constant, we have the proportionality 
#
# $p(\text{class}|\text{tokens}) \propto p(\text{tokens}|\text{class})\cdot p(\text{class})$
#
# The left hand side of the above expression is called the $\textbf{posterior class probability}$, the probability that the review is positive (or negative), given the tokens it contains. This is exactly what we want to predict!

# %% [markdown]
# ### 9D.2 The Naive Bayes Classifier
#
# #### Given the list of tokens in a review, we seek to predict whether the review is rated as `positive` or `negative` 
#
# #### We can make the prediction if we know the `posterior class probabilities`.
#
# #### $p(\text{class}|\text{tokens})$,
# #### where $\text{class}$ is either `positive` or `negative`, and $\text{tokens}$ is the list of tokens that appear in the review.
# #### [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) tells us that the posterior probabilities, the likelihoods and the priors are related this way:
#
# #### $p(\text{class}|\text{tokens}) \propto p(\text{tokens}|\text{class})\cdot p(\text{class})$
#
# #### Now the tokens are not independent of one another.  For example, 'go' often appears with 'to', so if 'go' appears in a review it is more likely that the review also contains 'to'. Nevertheless, assuming the tokens are independent allows us to simplify things, so we recklessly do it, hoping it's not too wrong!
# #### $p(\text{tokens}|\text{class}) = \prod_{i=1}^{n} p(t_{i}|\text{class})$
#
# #### where $t_{i}$ is the $i\text{th}$ token in the vocabulary and $n$ is the number of tokens in the vocabulary. 
#
# #### So Bayes' theorem is
#
# #### $p(\text{class}|\text{tokens}) \propto p(\text{class}) \prod_{i=1}^{n} p(t_{i}|\text{class}) $
#
# #### Taking the ratio of the $\textbf{posterior class probabilities}$ for the `positive` and `negative` classes, we have
#
# #### $\frac{p(+|\text{tokens})}{p( - |\text{tokens})} =  \frac{p(+)}{p( - )}  \cdot  \prod_{i=1}^{n} \frac {p(t_{i}|+)}  {p(t_{i}| - )} = \frac{p}{q}  \cdot  \prod_{i=1}^{n} \frac {L(t_{i}|+)}  {L(t_{i}| - )}$
# #### since likelihoods are proportional to probabilities.
# #### Taking the log of both sides converts this to a `linear` problem:
# #### $\text{log} \frac{p(+|\text{tokens})}{p( - |\text{tokens})} = \text{log}\frac{p}{q} + \sum_{i=1}^{n} \text{log} \frac {L(t_{i}|+)}  {L(t_{i}| - )} = b + \sum_{i=1}^{n}  R_{t_{i}}$
#
# #### The first term on the right-hand side is the `bias`, and the second term is the dot product of the *binarized* embedding vector and the log-count ratios
#
# #### If the left-hand side is greater than or equal to zero, we predict the review is `positive`, else we predict the review is `negative`. 
#
# ####  We can re-write the last equation in matrix form to generate a $m \times 1$ boolean column vector $\textbf{preds}$ of review predictions:
#
# #### $\textbf{preds} = \textbf{W} \cdot \textbf{R} + \textbf{b}$
# #### where 
#
# * $\textbf{preds} \equiv \text{log} \frac{p(+|\text{tokens})}{p( - |\text{tokens})}$
# * $\textbf{W}$ is the $m\times n$ `binarized document-term matrix`, whose rows are the binarized embedding vectors for the movie reviews
# * $\textbf{R}$ is the $n\times 1$ vector of `log-count ratios`  for the tokens, and 
# * $\textbf{b}$ is a $n\times 1$ vector whose entries are the bias $b$
#
#
# #### The Naive Bayes model consists of the log-counts vector $\textbf{R}$ and the bias $\textbf{b}$

# %% [markdown]
# ### 9E. Implement our Naive Bayes Movie Review classifier
# #### and use it to predict labels for the training and validation sets of the IMDb_sample data.

# %%
W = train_doc_term.sign()
preds_train = (W @ R + b) > 0
train_accuracy = (preds_train == y.items).mean()
print(f'The prediction accuracy for the training set is {train_accuracy}')

# %%
W = valid_doc_term.sign()
preds_valid = (W @ R + b) > 0
valid_accuracy = (preds_valid == valid_y.items).mean()
print(f'The prediction accuracy for the validation set is {valid_accuracy}')

# %% [markdown]
# ### 9F. Summary: A recipe for the Naive Bayes  Classifier
# #### Here is a summary of our procedure for predicting labels with the Naive Bayes Classifier, starting with the training set `x` and the training labels `y`
#
#
# #### 1. Compute the token count vectors
# > C0 = np.squeeze(np.asarray(x[y.items==negative].sum(0))) <br> 
# > C1 = np.squeeze(np.asarray(x[y.items==positive].sum(0))) <br> 
#
# #### 2. Compute the token class likelihood vectors
# > L0 = (C0+1) / ((y.items==negative).sum() + 1) <br> 
# > L1 = (C1+1) / ((y.items==positive).sum() + 1) <br> 
#
# #### 3. Compute the log-count ratios vector
# > R = np.log(L1/L0)
#
# #### 4. Compute the bias term
# > b = np.log((y.items==positive).mean() / (y.items==negative).mean())
#
# #### 5. The Naive Bayes model consists of the log-counts vector $\textbf{R}$ and the bias $\textbf{b}$
# #### 6. Predict the movie review labels from a linear transformation of the log-count ratios vector:
# > preds = (W @ R + b) > 0, <br> 
# > where the weights matrix W = valid_doc_term.sign() is the binarized `valid_doc_term matrix` whose rows are the binarized embedding vectors for the movie reviews for which you want to predict ratings.
#

# %% [markdown]
# ## 10. Working with the full IMDb data set

# %% [markdown]
# Now that we have our approach working on a smaller sample of the data, we can try using it on the full dataset.

# %% [markdown]
# ### 10A. Download the data

# %%
path = untar_data(URLs.IMDB)
path.ls()

# %%
(path/'train').ls()

# %% [markdown]
# ### 10B. Preprocess the data

# %% [markdown]
# #### Attempt to split and label the data fails most of the time, throwing a `BrokenProcessPool`  error; we apply a `brute force` approach, trying repeatedly until we succeed. Takes 10 minutes if it goes on the first try.

# %%
# %%time
# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!
count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        reviews_full = (TextList.from_folder(path)
             #  Make a `TextList` object that is a list of `WindowsPath` objects, 
             #     each of which contains the full path to one of the data files.
             .split_by_folder(valid='test')
             # Generate a `LabelLists` object that splits files by training and validation folders
             # Note: .label_from_folder in next line causes the `BrokenProcessPool` error
             .label_from_folder(classes=['neg', 'pos']))
             # Create a `CategoryLists` object which contains the data and
             #   its labels that are derived from folder names
        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')


# %% [markdown]
# ### 10C. Create document-term matrices for training and validation sets. 
# #### This takes about ~4 sec per matrix

# %%
# %%time
valid_doc_term = get_doc_term_matrix(reviews_full.valid.x, len(reviews_full.vocab.itos))

# %%
# %%time
train_doc_term = get_doc_term_matrix(reviews_full.train.x, len(reviews_full.vocab.itos))

# %% [markdown]
# ### 10D. Save the data
# When storing data like this, always make sure it's included in your `.gitignore` file

# %%
scipy.sparse.save_npz("train_doc_term.npz", train_doc_term)

# %%
scipy.sparse.save_npz("valid_doc_term.npz", valid_doc_term)

# %%
with open('reviews_full.pickle', 'wb') as handle:
    pickle.dump(reviews_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown]
# #### In the future, we'll just be able to load our data:

# %%
train_doc_term = scipy.sparse.load_npz("train_doc_term.npz")
valid_doc_term = scipy.sparse.load_npz("valid_doc_term.npz")

# %%
with open('reviews_full.pickle', 'rb') as handle:
    pickle.load(handle)

# %% [markdown]
# ## 11. Understanding Fastai's API$^\dagger$ for text data sets <br>
# $^\dagger$API $\equiv$ Application Programming Interface

# %% [markdown]
# #### reviews_full is a `LabelLists` object, which contains `LabelList` objects `train`, `valid` and potentially `test`

# %%
type(reviews_full)

# %%
type(reviews_full.valid)

# %% [markdown]
# #### reviews_full also contains the `vocab` object though it is not shown with the dir() command. This is an error.

# %%
print(reviews_full.vocab)

# %% [markdown]
# #### We will store the `vocabulary` in a variable `full_vocab`

# %%
full_vocab = reviews_full.vocab

# %% [markdown]
# #### Recall that a `vocab` object has a method `itos` which returns a list of tokens

# %%
full_vocab.itos[100:110]

# %% [markdown]
# #### A LabelList object contains a `TextList` object `x` and a `CategoryList` object `y` 

# %%
reviews_full.valid

# %% [markdown]
# #### A `TextList` object is a list of `Text` objects containing the reviews as items

# %%
type(reviews_full.valid.x[0])

# %% [markdown]
# #### A `Text` object has properties 
# #### `text`, which is a `str` containing the review text:

# %%
reviews_full.valid.x[0].text

# %% [markdown]
# ####  and  `data`,  which is an array of integers representing the tokens in the review:

# %%
reviews_full.valid.x[0].data

# %% [markdown]
# #### A `Text` object also has a method `.items` which returns the integer array representations for all the reviews

# %%
reviews_full.valid.x.items

# %% [markdown]
# #### Review labels are stored as a `CategoryList` object

# %%
type(reviews_full.valid.y)

# %% [markdown]
# #### A `CategoryList` object is a list of `Category` objects

# %%
type(reviews_full.valid.y[0])

# %%
reviews_full.valid.y[0]

# %% [markdown]
# #### A `Category` object also has a method `.items` which returns an array of integers labels for all the reviews

# %%
reviews_full.valid.y.items

# %% [markdown]
# #### The label of the first review seems right

# %%
reviews_full.valid.y[0]

# %% [markdown]
# #### Names of classes

# %%
reviews_full.valid.y.classes

# %% [markdown]
# #### Number of classes

# %%
reviews_full.valid.y.c

# %% [markdown]
# #### The classes have both integer rand string representations:

# %%
reviews_full.valid.y.c2i

# %%
reviews_full.valid.y[0].data

# %%
reviews_full.valid.y[0].obj

# %% [markdown]
# #### The training and validation data sets each have 25000 samples

# %%
len(reviews_full.train), len(reviews_full.valid)

# %% [markdown]
# ## 12. The Naive Bayes classifier with the full IMDb dataset

# %%
x=train_doc_term
y=reviews_full.train.y
valid_y = reviews_full.valid.y.items

# %%
x

# %%
positive = y.c2i['pos']
negative = y.c2i['neg']

# %%
C0 = np.squeeze(np.asarray(x[y.items==negative].sum(0)))
C1 = np.squeeze(np.asarray(x[y.items==positive].sum(0)))

# %%
C0

# %%
C1

# %% [markdown]
# ### 12A. Data exploration: log-count ratios

# %% [markdown]
# #### Token likelihoods conditioned on class

# %%
L1 = (C1+1) / ((y.items==positive).sum() + 1)
L0 = (C0+1) / ((y.items==negative).sum() + 1)

# %% [markdown]
# #### log-count ratios

# %%
R = np.log(L1/L0)

# %% [markdown]
# #### Examples of log-count ratios for a few words
# Check that log-count ratios are negative for words with `negative` sentiment and positive for words with `positive` sentiment! 

# %%
R[full_vocab.stoi['hated']]

# %%
R[full_vocab.stoi['loved']]

# %%
R[full_vocab.stoi['liked']]

# %%
R[full_vocab.stoi['worst']]

# %%
R[full_vocab.stoi['best']]

# %% [markdown]
# #### Since we have equal numbers of positive and negative reviews in this data set, the `bias` $b$ is 0.

# %%
b = np.log((y.items==positive).mean() / (y.items==negative).mean())
print(f'The bias term b is {b}')

# %% [markdown]
# ### 12B. Predictions of the Naive Bayes Classifier for the full IMDb data set.
# #### We get much better accuracy this time, because of the larger training set.

# %%
# predict labels for the validation data
W = valid_doc_term.sign()
preds = (W @ R + b) > 0
valid_accuracy = (preds == valid_y).mean()
print(f'Validation accuracy is {valid_accuracy} for the full data set')

# %% [markdown]
# ## 13. The Logistic Regression classifier with the full IMBb data set

# %% [markdown]
# #### With the `sci-kit learn` library, we can fit logistic a regression model where the features are the unigrams. Here $C$ is a regularization parameter.

# %%
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# #### Using the full `document-term matrix`:

# %%
m = LogisticRegression(C=0.1, dual=False,solver = 'liblinear')
# 'liblinear' and 'newton-cg' solvers both get 0.88328 accuracy
# 'sag', 'saga', and 'lbfgs' don't converge
m.fit(train_doc_term, y.items.astype(int))
preds = m.predict(valid_doc_term)
valid_accuracy = (preds==valid_y).mean()
print(f'Validation accuracy is {valid_accuracy} using the full doc-term matrix')

# %% [markdown]
# #### Using the binarized `document-term` matrix gets a slightly higher accuracy:

# %%
m = LogisticRegression(C=0.1, dual=False,solver = 'liblinear')
m.fit(train_doc_term.sign(), y.items.astype(int))
preds = m.predict(valid_doc_term.sign())
valid_accuracy = (preds==valid_y).mean()
print(f'Validation accuracy is {valid_accuracy} using the binarized doc-term matrix')

# %% [markdown]
# ## 14. `Trigram` representation of the `IMDb_sample`: preprocessing

# %% [markdown]
# #### Our next model is a version of logistic regression with Naive Bayes features extended to include bigrams and trigrams as well as unigrams, described [here](https://www.aclweb.org/anthology/P12-2018). For every document we compute binarized features as described above, but this time we use bigrams and trigrams too. Each feature is a log-count ratio. A logistic regression model is then trained to predict sentiment. Because of the much larger number of features, we will return to the smaller `IMDb_sample` data set.

# %% [markdown]
# ### What are `ngrams`?

# %% [markdown]
# #### An `n-gram` is a contiguous sequence of n items (where the items can be characters, syllables, or words).  A `1-gram` is a `unigram`, a `2-gram` is a `bigram`, and a `3-gram` is a `trigram`.
#
# #### Here, we are referring to sequences of words. So examples of bigrams include "the dog", "said that", and "can't you".

# %% [markdown]
# ### 14A. Get the IMDb_sample

# %%
path = untar_data(URLs.IMDB_SAMPLE)

# %% [markdown]
# ####  Again we find that accessing the `TextList` API *sometimes* (about 50% of the time) throws a `BrokenProcessPool` Error. This is puzzling, I don't know why it happens. But usually works on 1st or 2nd try.

# %%
# %%time
# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!

count = 0
error = True
while error:
    try: 
        # Preprocessing steps
        movie_reviews = (TextList.from_csv(path, 'texts.csv', cols='text')
                .split_from_df(col=2)
                .label_from_df(cols=0))

        error = False
        print(f'failure count is {count}\n')    
    except: # catch *all* exceptions
        # accumulate failure count
        count = count + 1
        print(f'failure count is {count}')


# %% [markdown]
# #### IMDb_sample vocabulary

# %%
vocab_sample = movie_reviews.vocab.itos
vocab_len = len(vocab_sample)
print(f'IMDb_sample vocabulary has {vocab_len} tokens')

# %% [markdown]
# ### 14B. Create the `ngram-doc matrix` for the training data

# %% [markdown]
# #### Just as the `doc-term matrix` encodes the `token` features, the `ngram-doc matrix` encodes the `ngram` features.

# %%
min_n=1
max_n=3

j_indices = []
indptr = []
values = []
indptr.append(0)
num_tokens = vocab_len

itongram = dict()
ngramtoi = dict()

# %% [markdown]
# #### We will iterate through the sequences of words to create our n-grams. This takes several minutes:

# %%
# %%time
for i, doc in enumerate(movie_reviews.train.x):
    feature_counter = Counter(doc.data)
    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    this_doc_ngrams = list()

    m = 0
    for n in range(min_n, max_n + 1):
        for k in range(vocab_len - n + 1):
            ngram = doc.data[k: k + n]
            if str(ngram) not in ngramtoi:
                if len(ngram)==1:
                    num = ngram[0]
                    ngramtoi[str(ngram)] = num
                    itongram[num] = ngram
                else:
                    ngramtoi[str(ngram)] = num_tokens
                    itongram[num_tokens] = ngram
                    num_tokens += 1
            this_doc_ngrams.append(ngramtoi[str(ngram)])
            m += 1

    ngram_counter = Counter(this_doc_ngrams)
    j_indices.extend(ngram_counter.keys())
    values.extend(ngram_counter.values())
    indptr.append(len(j_indices))

# %% [markdown]
# #### Using dictionaries to convert between indices and strings (in this case, for n-grams) is a common and useful approach!  Here, we have created `itongram` (index to n-gram) and `ngramtoi` (n-gram to index) dictionaries. This takes a few minutes...

# %%
# %%time
train_ngram_doc_matrix = scipy.sparse.csr_matrix((values, j_indices, indptr),
                                   shape=(len(indptr) - 1, len(ngramtoi)),
                                   dtype=int)

# %%
train_ngram_doc_matrix

# %% [markdown]
# ### 14C. Examine some ngrams in the training data

# %%
len(ngramtoi), len(itongram)

# %%
itongram[20005]

# %%
ngramtoi[str(itongram[20005])]

# %%
vocab_sample[125],vocab_sample[340],vocab_sample[10], 

# %%
itongram[100000]

# %%
vocab_sample[42], vocab_sample[49]

# %%
itongram[100010]

# %%
vocab_sample[38], vocab_sample[862]

# %%
itongram[6116]

# %%
vocab_sample[867], vocab_sample[52], vocab_sample[5]

# %%
itongram[6119]

# %%
vocab_sample[3376], vocab_sample[5], vocab_sample[1800]

# %%
itongram[80000]

# %%
vocab_sample[0], vocab_sample[1240], vocab_sample[0]

# %% [markdown]
# ### 14D. Create the `ngram-doc matrix` for the validation data

# %%
# %%time
j_indices = []
indptr = []
values = []
indptr.append(0)

for i, doc in enumerate(movie_reviews.valid.x):
    feature_counter = Counter(doc.data)
    j_indices.extend(feature_counter.keys())
    values.extend(feature_counter.values())
    this_doc_ngrams = list()

    m = 0
    for n in range(min_n, max_n + 1):
        for k in range(vocab_len - n + 1):
            ngram = doc.data[k: k + n]
            if str(ngram) in ngramtoi:
                this_doc_ngrams.append(ngramtoi[str(ngram)])
            m += 1

    ngram_counter = Counter(this_doc_ngrams)
    j_indices.extend(ngram_counter.keys())
    values.extend(ngram_counter.values())
    indptr.append(len(j_indices))

# %%
# %%time
valid_ngram_doc_matrix = scipy.sparse.csr_matrix((values, j_indices, indptr),
                                   shape=(len(indptr) - 1, len(ngramtoi)),
                                   dtype=int)

# %%
valid_ngram_doc_matrix

# %% [markdown]
# ### 14E. Save the `ngram` data so we won't have to spend the time to generate it again

# %%
scipy.sparse.save_npz("train_ngram_matrix.npz", train_ngram_doc_matrix)
scipy.sparse.save_npz("valid_ngram_matrix.npz", valid_ngram_doc_matrix)

# %%
with open('itongram.pickle', 'wb') as handle:
    pickle.dump(itongram, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('ngramtoi.pickle', 'wb') as handle:
    pickle.dump(ngramtoi, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% [markdown] heading_collapsed=true
# ### 14F. Load the `ngram` data

# %% hidden=true
train_ngram_doc_matrix = scipy.sparse.load_npz("train_ngram_matrix.npz")
valid_ngram_doc_matrix = scipy.sparse.load_npz("valid_ngram_matrix.npz")

# %% hidden=true
with open('itongram.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
with open('ngramtoi.pickle', 'rb') as handle:
    b = pickle.load(handle)

# %% [markdown]
# ## 15. A Naive Bayes IMDb classifier using Trigrams instead of Tokens

# %%
x=train_ngram_doc_matrix
x

# %%
k = x.shape[1]
print(f'There are {k} 1-gram, 2-gram, and 3-gram features in the IMDb_sample vocabulary')

# %%
y=movie_reviews.train.y
y.items
y.items.shape

# %% [markdown]
# #### Numerical label representation

# %%
positive = y.c2i['positive']
negative = y.c2i['negative']
print(f'positive and negative review labels are represented numerically by {positive} and {negative}')

# %% [markdown]
# #### Boolean indicator tells whether or not a training label is positive

# %%
valid_labels = [label == positive for label in movie_reviews.valid.y.items]
valid_labels=np.array(valid_labels)[:,np.newaxis]
valid_labels.shape

# %% [markdown]
# #### Boolean indicators for `positive` and `negative` reviews in the training set

# %%
pos = (y.items == positive)
neg = (y.items == negative)

# %% [markdown]
# ### 15A. Naive Bayes with Trigrams

# %% [markdown]
# #### The input is the full `ngram_doc_matrix`

# %% [markdown]
# #### Token `occurrence count` vectors
# The kernel dies if I use the sparse matrix x here, so converting x to a dense matrix

# %%
C0 = np.squeeze(x.todense()[neg].sum(0))
C1 = np.squeeze(x.todense()[pos].sum(0))

# %% [markdown]
# #### Token `class likelihood` vectors

# %%
L0 = (C0+1) / (neg.sum() + 1)
L1 = (C1+1) / (pos.sum() + 1)

# %% [markdown]
# #### `log-count ratio` column vector

# %%
R = np.log(L1/L0).reshape((-1,1))

# %% [markdown]
# #### bias

# %%
(y.items==positive).mean(), (y.items==negative).mean()

# %%
b = np.log((y.items==positive).mean() / (y.items==negative).mean())
print(b)

# %% [markdown]
# #### The input is the  `ngram_doc_matrix`

# %%
W = valid_ngram_doc_matrix

# %% [markdown]
# #### Label predictions with the full ngram_doc_matrix

# %%
preds = W @ R + b
preds = preds > 0

# %% [markdown]
# #### Accuracy is much better than with the unigram model

# %%
accuracy = (preds == valid_labels).mean()
print(f'Accuracy for Naive Bayes with the full trigrams Model = {accuracy}' )

# %% [markdown]
# ### 15B. Binarized Naive Bayes with Trigrams

# %% [markdown]
# #### The input data is the binarized `n_gram_doc_matrix`

# %%
x = train_ngram_doc_matrix.sign()
x

# %% [markdown]
# #### Token `occurrence count` vectors
# The kernel dies if I use the sparse matrix x here, so converting x to a dense matrix

# %%
C0 = np.squeeze(x.todense()[neg].sum(0))
C1 = np.squeeze(x.todense()[pos].sum(0))

# %% [markdown]
# #### Token `class likelihood` vectors

# %%
L1 = (C1+1) / ((y.items==positive).sum() + 1)
L0 = (C0+1) / ((y.items==negative).sum() + 1)

# %% [markdown]
# #### `log-count ratio` column vector

# %%
R = np.log(L1/L0).reshape((-1,1))
print(R)

# %% [markdown]
# #### Input to the model is the binarized `ngram_doc_matrix`

# %%
W = valid_ngram_doc_matrix.sign()

# %% [markdown]
# #### Label predictions with the binarized ngram_doc_matrix

# %%
preds = W @ R + b
preds = preds>0

# %% [markdown]
# #### Accuracy is still much better than with unigram model, but this time a bit worse with the binarized model

# %%
accuracy = (preds==valid_labels).mean()
print(f'Accuracy for Binarized Naive Bayes with Trigrams Model = {accuracy}' )

# %% [markdown]
# ## 16. A Logistic Regression IMDb classifier using Trigrams

# %% [markdown]
# #### Here we fit `regularized` logistic regression where the features are the trigrams.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# %% [markdown]
# ### 16A. Use `CountVectorizer` to create the `train_ngram_doc` matrix

# %%
veczr = CountVectorizer(ngram_range=(1,3), preprocessor=noop, tokenizer=noop, max_features=800000)

# %%
train_docs = movie_reviews.train.x
train_words = [[movie_reviews.vocab.itos[o] for o in doc.data] for doc in train_docs]

# %%
valid_docs = movie_reviews.valid.x
valid_words = [[movie_reviews.vocab.itos[o] for o in doc.data] for doc in valid_docs]

# %%
# %%time
train_ngram_doc_matrix_veczr = veczr.fit_transform(train_words)
train_ngram_doc_matrix_veczr

# %%
valid_ngram_doc_matrix_veczr = veczr.transform(valid_words)
valid_ngram_doc_matrix_veczr

# %%
vocab = veczr.get_feature_names()

# %%
vocab[200000:200005]

# %% [markdown]
# #### Binarized trigram counts

# %%
# fit model
m = LogisticRegression(C=0.1, dual=False, solver = 'liblinear')
m.fit(train_ngram_doc_matrix_veczr.sign(), y.items);

# get predictions
preds = m.predict(valid_ngram_doc_matrix_veczr.sign())
valid_labels = [label == positive for label in movie_reviews.valid.y.items]

# check accuracy
accuracy = (preds==valid_labels).mean()
print(f'Accuracy = {accuracy} for Logistic Regression, with binarized trigram counts from `CountVectorizer`' )

# %% [markdown]
# #### Full trigram counts
# Performance is worse with full trigram counts.

# %%
m = LogisticRegression(C=0.1, dual=False, solver = 'liblinear')
m.fit(train_ngram_doc_matrix_veczr, y.items);

preds = m.predict(valid_ngram_doc_matrix_veczr)
accuracy =(preds==valid_labels).mean()
print(f'Accuracy  = {accuracy} for Logistic Regression, with full trigram counts from `CountVectorizer`' )

# %% [markdown]
# ### 16B. This time, use `our` ngrams to create the `train_ngram_doc` matrix

# %%
train_ngram_doc_matrix.shape

# %% [markdown]
# #### Fit a model to the binarized trigram counts

# %%
m2=None
m2 = LogisticRegression(C=0.1, dual=False, solver = 'liblinear')
m2.fit(train_ngram_doc_matrix.sign(), y.items)

preds = m2.predict(valid_ngram_doc_matrix.sign())
accuracy = (preds==valid_labels).mean()
print(f'Accuracy  = {accuracy} for Logistic Regression, with our binarized trigram counts' )

# %% [markdown]
# #### Fit a model to the full trigram counts
# Performance is again worse with full trigram counts.

# %%
m2 = LogisticRegression(C=0.1, dual=False,solver='liblinear')
m2.fit(train_ngram_doc_matrix, y.items)
preds = m2.predict(valid_ngram_doc_matrix)
accuracy = (preds==valid_labels).mean()
print(f'Accuracy  = {accuracy} for Not-Binarized Logistic Regression, with our Trigrams' )

# %% [markdown]
# ### 16C. Logistic Regression with the log-count ratio gives a slightly better result

# %% [markdown]
# #### Compute the $\text{log-count ratio}, \textbf{R}$  and the $\text{bias}, \textbf{b}$

# %%
x=train_ngram_doc_matrix.sign()
valid_x=valid_ngram_doc_matrix.sign()

# %%
C0 = np.squeeze(x.todense()[neg].sum(axis=0))
C1 = np.squeeze(x.todense()[pos].sum(axis=0))

# %%
L1 = (C1+1) / ((pos).sum() + 1)
L0 = (C0+1) / ((neg).sum() + 1)

# %%
R = np.log(L1/L0)
R.shape

# %% [markdown]
# #### Here we fit regularized logistic regression where the features are the log-count ratios for the trigrams':

# %%
R_tile = np.tile(R,[x.shape[0],1])
print(R_tile.shape)

# %%
# The next line causes the kernel to die?
# x_nb = x.multiply(R)
# As a workaround, use the full matrices
x_nb = np.multiply(x.todense(),R_tile)
m = LogisticRegression(dual=False, C=0.1,solver='liblinear')
m.fit(x_nb, y.items);

# why does valid_x.multiply(R) work but x.multiply(R) does not?
valid_x_nb = valid_x.multiply(R) 
preds = m.predict(valid_x_nb)

accuracy = (preds==valid_labels).mean()
print(f'Accuracy  = {accuracy} for Logistic Regression, with trigram log-count ratios' )

# %% [markdown]
# ## 17. Summary of movie review sentiment classifier results

# %%
from IPython.display import HTML, display
# Note: to install the `tabulate` package, 
#     go to a shell terminal and run the command
#     `conda install tabulate`
import tabulate
table = [["Model","Data Set","Token Unit","Validation Accuracy(%)"],
         ["Naive Bayes","IMDb_sample", "Full Unigram","64.5 (from video #5)"],
         ["Naive Bayes","IMDb_sample", "Binarized Unigram","68.0"],
         ["Naive Bayes","IMDb_sample", "Full Trigram","76.0"],
         ["Naive Bayes","IMDb_sample", "Binarized Trigram","73.5"],
         ["Logistic Regression","IMDb_sample", "Full Trigram","78.0, 80.0 (our Trigrams)"],
         ["Logistic Regression","IMDb_sample", "Binarized Trigram","83.0"],
         ["Logistic Regression","IMDb_sample", "Binarized Trigram log-count ratios","83.5"],
         ["Naive Bayes","Full IMDb","IMDb_sample", "Binarized Trigram","83.3"],
         ["Logistic Regression","Full IMDb", "Full Trigram","88.3"],
         ["Logistic Regression","Full IMDb", "Binarized Trigram","88.5"]]
display(HTML(tabulate.tabulate(table, tablefmt='html')))

# %% [markdown]
# ## References

# %% [markdown]
# * Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. Sida Wang and Christopher D. Manning [pdf](https://www.aclweb.org/anthology/P12-2018)
# * [The Naive Bayes Classifier](https://towardsdatascience.com/the-naive-bayes-classifier-e92ea9f47523). Joseph Catanzarite, in Towards Data Science

# %% [markdown]
#
