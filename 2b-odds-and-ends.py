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
# In this notebook, I want to wrap up some loose ends from last time.

# %% [markdown]
# ## The two cultures

# %% [markdown]
# This "debate" captures the tension between two approaches:
#
# - modeling the underlying mechanism of a phenomena
# - using machine learning to predict outputs (without necessarily understanding the mechanisms that create them)

# %% [markdown]
# <img src="images/glutathione.jpg" alt="One carbon cell metabolism" style="width: 80%"/>

# %% [markdown]
# I was part of a research project (in 2007) that involved manually coding each of the above reactions.  We were determining if the final system could generate the same ouputs (in this case, levels in the blood of various substrates) as were observed in clinical studies.  
#
# The equation for each reaction could be quite complex:
# <img src="images/vcbs.png" alt="reaction equation" style="width: 80%"/>
#
# This is an example of modeling the underlying mechanism, and is very different from a machine learning approach.
#
# Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2391141/

# %% [markdown]
# ## The most popular word in each state

# %% [markdown]
# <img src="images/map-popular-word.png" alt="The" style="width: 80%"/>

# %% [markdown]
# A time to remove stop words

# %% [markdown]
# ## Factorization is analgous to matrix decomposition

# %% [markdown] heading_collapsed=true
# ### With Integers

# %% [markdown] hidden=true
# Multiplication: 
# 	$$2 * 2 * 3 * 3 * 2 * 2 \rightarrow 144$$
#     
# <img src="images/factorization.png" alt="factorization" style="width: 50%"/>
#
# Factorization is the “opposite” of multiplication: 
# 	 $$144 \rightarrow 2 * 2 * 3 * 3 * 2 * 2$$
#      
# Here, the factors have the nice property of being prime.
#
# Prime factorization is much harder than multiplication (which is good, because it’s the heart of encryption).

# %% [markdown] heading_collapsed=true
# ### With Matrices

# %% [markdown] hidden=true
# Matrix decompositions are a way of taking matrices apart (the "opposite" of matrix multiplication).
#
# Similarly, we use matrix decompositions to come up with matrices with nice properties.

# %% [markdown] hidden=true
# Taking matrices apart is harder than putting them together.
#
# [One application](https://github.com/fastai/numerical-linear-algebra/blob/master/nbs/3.%20Background%20Removal%20with%20Robust%20PCA.ipynb):
#
# <img src="images/grid1.jpg" alt="The" style="width: 100%"/>

# %% [markdown] hidden=true
# What are the nice properties that matrices in an SVD decomposition have?
#
# $$A = USV$$

# %% [markdown]
# ## Some Linear Algebra Review

# %% [markdown]
# ### Matrix-vector multiplication

# %% [markdown]
# $Ax = b$ takes a linear combination of the columns of $A$, using coefficients $x$
#
# http://matrixmultiplication.xyz/

# %% [markdown]
# ### Matrix-matrix multiplication

# %% [markdown]
# $A B = C$ each column of C is a linear combination of columns of A, where the coefficients come from the corresponding column of C

# %% [markdown]
# <img src="images/face_nmf.png" alt="NMF on faces" style="width: 80%"/>
#
# (source: [NMF Tutorial](http://perso.telecom-paristech.fr/~essid/teach/NMF_tutorial_ICME-2014.pdf))

# %% [markdown]
# ### Matrices as Transformations

# %% [markdown]
# The 3Blue 1Brown [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) videos are fantastic.  They give a much more visual & geometric perspective on linear algreba than how it is typically taught.  These videos are a great resource if you are a linear algebra beginner, or feel uncomfortable or rusty with the material.
#
# Even if you are a linear algrebra pro, I still recommend these videos for a new perspective, and they are very well made.

# %%
from IPython.display import YouTubeVideo

YouTubeVideo("kYB8IZa5AuE")

# %% [markdown]
# ## British Literature SVD & NMF in Excel

# %% [markdown]
# Data was downloaded from [here](https://de.dariah.eu/tatom/datasets.html)
#
# The code below was used to create the matrices which are displayed in the SVD and NMF of British Literature excel workbook. The data is intended to be viewed in Excel, I've just included the code here for thoroughness.

# %% [markdown] heading_collapsed=true
# ### Initializing, create document-term matrix

# %% hidden=true
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import decomposition
from glob import glob
import os

# %% hidden=true
np.set_printoptions(suppress=True)

# %% hidden=true
filenames = []
for folder in ["british-fiction-corpus"]: #, "french-plays", "hugo-les-misérables"]:
    filenames.extend(glob("data/literature/" + folder + "/*.txt"))

# %% hidden=true
len(filenames)

# %% hidden=true
vectorizer = TfidfVectorizer(input='filename', stop_words='english')
dtm = vectorizer.fit_transform(filenames).toarray()
vocab = np.array(vectorizer.get_feature_names())
dtm.shape, len(vocab)

# %% hidden=true
[f.split("/")[3] for f in filenames]

# %% [markdown] heading_collapsed=true
# ### NMF

# %% hidden=true
clf = decomposition.NMF(n_components=10, random_state=1)

W1 = clf.fit_transform(dtm)
H1 = clf.components_

# %% hidden=true
num_top_words=8

def show_topics(a):
    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]
    topic_words = ([top_words(t) for t in a])
    return [' '.join(t) for t in topic_words]


# %% hidden=true
def get_all_topic_words(H):
    top_indices = lambda t: {i for i in np.argsort(t)[:-num_top_words-1:-1]}
    topic_indices = [top_indices(t) for t in H]
    return sorted(set.union(*topic_indices))


# %% hidden=true
ind = get_all_topic_words(H1)

# %% hidden=true
vocab[ind]

# %% hidden=true
show_topics(H1)

# %% hidden=true
W1.shape, H1[:, ind].shape

# %% [markdown] hidden=true
# #### Export to CSVs

# %% hidden=true
from IPython.display import FileLink, FileLinks

# %% hidden=true
np.savetxt("britlit_W.csv", W1, delimiter=",", fmt='%.14f')
FileLink('britlit_W.csv')

# %% hidden=true
np.savetxt("britlit_H.csv", H1[:,ind], delimiter=",", fmt='%.14f')
FileLink('britlit_H.csv')

# %% hidden=true
np.savetxt("britlit_raw.csv", dtm[:,ind], delimiter=",", fmt='%.14f')
FileLink('britlit_raw.csv')

# %% hidden=true
[str(word) for word in vocab[ind]]

# %% [markdown]
# ### SVD

# %%
U, s, V = decomposition.randomized_svd(dtm, 10)

# %%
ind = get_all_topic_words(V)

# %%
len(ind)

# %%
vocab[ind]

# %%
show_topics(H1)

# %%
np.savetxt("britlit_U.csv", U, delimiter=",", fmt='%.14f')
FileLink('britlit_U.csv')

# %%
np.savetxt("britlit_V.csv", V[:,ind], delimiter=",", fmt='%.14f')
FileLink('britlit_V.csv')

# %%
np.savetxt("britlit_raw_svd.csv", dtm[:,ind], delimiter=",", fmt='%.14f')
FileLink('britlit_raw_svd.csv')

# %%
np.savetxt("britlit_S.csv", np.diag(s), delimiter=",", fmt='%.14f')
FileLink('britlit_S.csv')

# %%
[str(word) for word in vocab[ind]]

# %% [markdown]
# ## Randomized SVD offers a speed up

# %% [markdown]
# <img src="images/svd_slow.png" alt="" style="width: 80%"/>

# %% [markdown]
# One way to address this is to use randomized SVD.  In the below chart, the error is the difference between A - U * S * V, that is, what you've failed to capture in your decomposition:
#
# <img src="images/svd_speed.png" alt="" style="width: 60%"/>

# %% [markdown]
# For more on randomized SVD, check out my [PyBay 2017 talk](https://www.youtube.com/watch?v=7i6kBz1kZ-A&list=PLtmWHNX-gukLQlMvtRJ19s7-8MrnRV6h6&index=7).
#
# For significantly more on randomized SVD, check out the [Computational Linear Algebra course](https://github.com/fastai/numerical-linear-algebra).

# %% [markdown]
# ## Full vs Reduced SVD

# %% [markdown]
# Remember how we were calling `np.linalg.svd(vectors, full_matrices=False)`?  We set `full_matrices=False` to calculate the reduced SVD.  For the full SVD, both U and V are **square** matrices, where the extra columns in U form an orthonormal basis (but zero out when multiplied by extra rows of zeros in S).

# %% [markdown]
# Diagrams from Trefethen:
#
# <img src="images/full_svd.JPG" alt="" style="width: 80%"/>
#
# <img src="images/reduced_svd.JPG" alt="" style="width: 70%"/>
