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
# Homework 2 is due on 6/18.  Please upload your completed assignment to Canvas as a .py script which consists of all the necessary functions.
#
# Name your script hw2_loginID.py For instance, I would name my script hw2_rlthomas3.py, since rlthomas3 is my USF login ID (it's what comes before @usfca.edu in my email address).

# %% [markdown]
# 1\. Consider this $5x5$ matrix :
#
# \begin{pmatrix}
#   1 & & & -2 & \\
#   & 3 & & & -9 \\
#   & & -7 & 4 & \\ 
#   -1 & 2 & 7 & & \\
#   -3 & & 26 & &
#  \end{pmatrix}
#  
#  a. Write how it would be stored in coordinate-wise format. Your answer should be a dictionary named `coo` with keys: `vals`, `cols`, and `rows`
#  
#  b. Write how it would be stored in compressed row format. Your answer should be a dictionary named `csr` with keys: `vals`, `cols`, and `row_ptr`

# %% [markdown]
# 2\. Write a method that uses regex:
#
# `get_dimensions("1280x720")` should return `1280, 720`

# %%
def get_dimensions(text):
    # something with regex
    return int(dim1), int(dim2)


# %% [markdown]
# 3\. Use regex to pick out the names of PDFs.
#
# `get_pdf_names("file_record_transcript.pdf")` should return `"file_record_transcript"`
#
# `get_pdf_names("testfile_fake.pdf.tmp")` should return `None`

# %%
def get_pdf_names(text):
    # something with regex
    
    return pdf_name

# %% [markdown]
# 4\. For each of the following, answer whether they are **parameters** or **activations**:
#
# - weights in a pre-trained network
# - hidden state in an RNN
# - attention weights
#
# Submit answer via this jot form: https://form.jotform.com/91605828322154
