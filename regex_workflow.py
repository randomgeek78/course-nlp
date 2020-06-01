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
#     display_name: fastai_v1
#     language: python
#     name: fastai_v1
# ---

# %% [markdown]
# # `regex` workflow

# %%
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
import pandas as pd
import re

# %% [markdown]
# ### Jeremy Howard is the guest lecturer for Lesson 9! <br>
#
# #### In the video, he gives a three-part lesson plan: 
#     * regex workflow
#     * svd
#     * transfer learning. 
#     
# Jeremy mentions that he uses `regex` every day in his work, and that it is essential for machine learning practitioners to develop a working knowledge of `regex`. Since we've already done deep dives into `svd` and into `transfer learning`, we'll focus on the `regex` part of this video, `from 1:50 to 21:29`.

# %% [markdown]
# ### A simple `regex` exercise
# #### To illustrate the power of `regex` and familiarize us with the way he works, Jeremy poses the following problem: <br>Let's extract all the phone numbers from the Austin Public Health Locations database and create a list of the phone numbers in the standard format `(ddd) ddd dddd`. He shows how to use `vim` to accomplish this task.
# Let's listen to Jeremy for the next 20 minutes or so:

# %%
from IPython.display import HTML

# Play youtube video
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/5gCQvuznKn0?start=110" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

# %% [markdown]
# #### Some of the takeaways from the video, paraphrased:
# 1. A neccessary but not sufficient condition for success<br>
# What is the greatest difference between people who succeed and people who do not? It's entirely about tenacity. If you are willing to focus on the task and keep trying you have a good chance of succeeding. 
#
# 2. Workflow<br>
# Work in an interactive environment, such as `vim`, or `jupyter notebook`, so you can try things get immediate feedback, revise, and progress toward a solution. 
#
# 3. Debugging<br>
# When your code fails, remember that the computer is doing exactly what you asked. A good general approach is to break the code up into smaller parts, then run it again, and find out which part doesn't work.
#
# 4. Humility<br>
# It's never "I think the problem in the code is X". A better approach is to start with the working assumption "I am an idiot, and I don't understand why things aren't working". Be willing to start from scratch and check every little step.

# %% [markdown]
# #### OK, let's get to work on our task. We'll use `jupyter notebook` as our interactive environment.

# %% [markdown]
# ## 1. Get the Austin Public Health Locations database
# #### https://data.austintexas.gov/Health-and-Community-Services/Austin-Public-Health-Locations/6v78-dj3u/data

# %%
path = 'C:/Users/cross-entropy/.fastai/data/Austin_Public_Health_Locations'

# %% [markdown]
# #### Read the data into a pandas dataframe. 
# From the `Phone Number` column, we see that the phone numbers are in the format `ddd-ddd-dddd`.

# %%
df = pd.read_csv(path+'/Austin_Public_Health_Locations.csv')
display(df)

# %% [markdown]
# #### Read the database into a raw text string. 
# This will be our starting point.

# %%
with open(path+'/Austin_Public_Health_Locations.csv', 'r') as file:
    data = file.read().replace('\n', '')
print(data)

# %% [markdown]
# ## 2. Extract the phone numbers

# %% [markdown]
# #### We first construct a regular expression to match the phone numbers and break them into tuples. This involved a bit of trial and error.

# %%
re_extract_phone_number = re.compile(r"(\d\d\d)-(\d+)-(\d+)")

# %%
phone_number_list = re_extract_phone_number.findall(data)
display(phone_number_list)

# %% [markdown]
# ## 3. Put the phone numbers in the desired format

# %% [markdown]
# #### Next we join together the tuples, separated by spaces:

# %%
[' '.join(tuple) for tuple in phone_number_list]

# %% [markdown]
# #### Voila! Finis.

# %%
