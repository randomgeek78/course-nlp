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
# # 0. Course Logistics

# %% [markdown]
# ## Ask Questions

# %% [markdown]
# Let me know how things are going. I've talked with your other professors, but I may not know everything you've seen/haven't seen.

# %% [markdown]
# ## Basics

# %% [markdown]
# **Office hours**: Email me if you need to meet.  I will be around on Friday afternoons (although meeting with practicum students for part of that time).
#
# My contact info: rachel@fast.ai
#
# Email me if you will need to miss class.
#
# Jupyter Notebooks will be available on Github at: https://github.com/fastai/course-nlp Please pull/download before class. Be sure to let me know **THIS WEEK** if you are having any problems running the notebooks from your own computer.  You may want to make a separate copy, because running Jupyter notebooks causes them to change, which can create github conflicts the next time you pull.

# %% [markdown]
# **Grading Rubric**:
#
# | Assignment         | Percent |
# |--------------------|:-------:|
# | Homework           |   35%   |
# | Writing assignment |   30%   |
# | Final Exam         |   35%   |

# %% [markdown]
# **Honor Code** 
#
# No cheating nor plagiarism is allowed, please see below for more details.

# %% [markdown]
# **On Laptops**
#
# I ask you to be respectful of me and your classmates and to refrain from surfing the web or using social media (facebook, twitter, etc) or messaging programs during class. It is absolutely forbidden to use instant messaging programs, email, etc. during class lectures or quizzes.

# %% [markdown]
# ## Syllabus

# %% [markdown]
# Topics Covered:
#
# 1\. What is NLP?
#   - A changing field
#   - Resources
#   - Tools
#   - Python libraries
#   - Example applications
#   - Ethics issues
#
# 2\. Topic Modeling with NMF and SVD
#   - Stop words, stemming, & lemmatization
#   - Term-document matrix
#   - Topic Frequency-Inverse Document Frequency (TF-IDF)
#   - Singular Value Decomposition (SVD)
#   - Non-negative Matrix Factorization (NMF)
#   - Truncated SVD, Randomized SVD
#
# 3\. Sentiment classification with Naive Bayes, Logistic regression, and ngrams
#   - Sparse matrix storage
#   - Counters
#   - the fastai library
#   - Naive Bayes
#   - Logistic regression
#   - Ngrams
#   - Logistic regression with Naive Bayes features, with trigrams
#   
# 4\. Regex (and re-visiting tokenization)
#
# 5\. Language modeling & sentiment classification with deep learning
#   - Language model
#   - Transfer learning
#   - Sentiment classification
#
# 6\. Translation with RNNs
#   - Review Embeddings
#   - Bleu metric
#   - Teacher Forcing
#   - Bidirectional
#   - Attention
#
# 7\. Translation with the Transformer architecture
#   - Transformer Model
#   - Multi-head attention
#   - Masking
#   - Label smoothing
#
# 8\. Bias & ethics in NLP
#   - bias in word embeddings
#   - types of bias
#   - attention economy
#   - drowning in fraudulent/fake info

# %% [markdown]
# ## Writing Assignment

# %% [markdown]
# **Writing Assignment:**  Writing about technical concepts is a hugely valuable skill.  I want you to write a technical blog post related to numerical linear algebra.  [A blog is like a resume, only better](http://www.fast.ai/2017/04/06/alternatives/). Technical writing is also important in creating documentation, sharing your work with co-workers, applying to speak at conferences, and practicing for interviews. (You don't actually have to publish it, although I hope you do, and please send me the link if you do.)

# %% [markdown]
# For topics, you can choose anything NLP-related (category of research, software library, algorithm, etc). 
#
# Feel free to ask me if you are wondering if your topic idea is suitable!
#
# Please read the following where I put together some advice and common pitfalls to avoid:
#
# [Advice for Better Blog Posts](https://www.fast.ai/2019/05/13/blogging-advice/)

# %% [markdown]
# ### Technical Blogs
#
# Great NLP-related blogs:
# - [Sebastian Ruder](http://ruder.io/)
# - [Jay Alammar](https://jalammar.github.io/)
# - [Abigail See](http://www.abigailsee.com/)
# - [Joyce Xu](https://medium.com/@joycex99)
# - [Stephen Merity](https://smerity.com/articles/articles.html)
# - [Rachael Tatman](https://towardsdatascience.com/evaluating-text-output-in-nlp-bleu-at-your-own-risk-e8609665a213)
#
# Other great technical blog posts:
# - [Peter Norvig](http://nbviewer.jupyter.org/url/norvig.com/ipython/ProbabilityParadox.ipynb) (more [here](http://norvig.com/ipython/))
# - [Julia Evans](https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture) (more [here](https://jvns.ca/blog/2014/08/12/what-happens-if-you-write-a-tcp-stack-in-python/))
# - [Julia Ferraioli](http://blog.juliaferraioli.com/2016/02/exploring-world-using-vision-twilio.html)
# - [Slav Ivanov](https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b)
# - find [more on twitter](https://twitter.com/math_rachel)

# %% [markdown]
# ## Deadlines

# %% [markdown]
# | Assignment        | Dates    |
# |-------------------|:--------:|
# | Homeworks         |   TBA    |
# | Writing           |   6/22   |
# | Final Exam        |   6/27   |

# %% [markdown]
# ## Teaching

# %% [markdown]
# **Teaching Approach**
#
# I'll be using a *top-down* teaching method, which is different from how most math courses operate.  Typically, in a *bottom-up* approach, you first learn all the separate components you will be using, and then you gradually build them up into more complex structures.  The problems with this are that students often lose motivation, don't have a sense of the "big picture", and don't know what they'll need.
#
# If you took the fast.ai deep learning course, that is what we used.  You can hear more about my teaching philosophy [in this blog post](http://www.fast.ai/2016/10/08/teaching-philosophy/) or [in this talk](https://vimeo.com/214233053).
#
# Harvard Professor David Perkins has a book, [Making Learning Whole](https://www.amazon.com/Making-Learning-Whole-Principles-Transform/dp/0470633719) in which he uses baseball as an analogy.  We don't require kids to memorize all the rules of baseball and understand all the technical details before we let them play the game.  Rather, they start playing with a just general sense of it, and then gradually learn more rules/details as time goes on.
#
# All that to say, don't worry if you don't understand everything at first!  You're not supposed to.  We will start using some "black boxes" or matrix decompositions that haven't yet been explained, and then we'll dig into the lower level details later.
#
# To start, focus on what things DO, not what they ARE.

# %% [markdown]
# People learn by:
# 1. **doing** (coding and building)
# 2. **explaining** what they've learned (by writing or helping others)

# %% [markdown] heading_collapsed=true
# ## USF Policies

# %% [markdown] hidden=true
# **Academic Integrity** 
#
# USF upholds the standards of honesty and integrity from all members of the academic community. All students are expected to know and adhere to the University’s Honor Code. You can find the full text of the [code online](www.usfca.edu/academic_integrity). The policy covers:
# - Plagiarism: intentionally or unintentionally representing the words or ideas of another person as your own; failure to properly cite references; manufacturing references.
# - Working with another person when independent work is required.
# - Submission of the same paper in more than one course without the specific permission of each instructor.
# - Submitting a paper written (entirely or even a small part) by another person or obtained from the internet.
# - Plagiarism is plagiarism: it does not matter if the source being copied is on the Internet, from a book or textbook, or from quizzes or problem sets written up by other students.
# - The penalties for violation of the policy may include a failing grade on the assignment, a failing grade in the course, and/or a referral to the Academic Integrity Committee.

# %% [markdown] hidden=true
# **Students with Disabilities**
#
# If you are a student with a disability or disabling condition, or if you think you may have a disability, please contact USF Student Disability Services (SDS) at 415 422-2613 within the first week of class, or immediately upon onset of disability, to speak with a disability specialist. If you are determined eligible for reasonable accommodations, please meet with your disability specialist so they can arrange to have your accommodation letter sent to me, and we will discuss your needs for this course. For more information, please visit [this website]( http://www.usfca.edu/sds) or call (415) 422-2613.

# %% [markdown] hidden=true
# **Behavioral Expectations**
#
# All students are expected to behave in accordance with the [Student Conduct Code and other University policies](https://myusf.usfca.edu/fogcutter). Open discussion and disagreement is encouraged when done respectfully and in the spirit of academic discourse. There are also a variety of behaviors that, while not against a specific University policy, may create disruption in this course. Students whose behavior is disruptive or who fail to comply with the instructor may be dismissed from the class for the remainder of the class period and may need to meet with the instructor or Dean prior to returning to the next class period. If necessary, referrals may also be made to the Student Conduct process for violations of the Student Conduct Code.

# %% [markdown] hidden=true
# **Counseling and Psychological Services**
#
# Our diverse staff offers brief individual, couple, and group counseling to student members of our community. CAPS services are confidential and free of charge. Call 415-422-6352 for an initial consultation appointment. Having a crisis at 3 AM? We are still here for you. Telephone consultation through CAPS After Hours is available between the hours of 5:00 PM to 8:30 AM; call the above number and press 2.

# %% [markdown] hidden=true
# **Confidentiality, Mandatory Reporting, and Sexual Assault**
#
# As an instructor, one of my responsibilities is to help create a safe learning environment on our campus. I also have a mandatory reporting responsibility related to my role as a faculty member. I am required to share information regarding sexual misconduct or information about a crime that may have occurred on USFs campus with the University. Here are other resources:
#
# - To report any sexual misconduct, students may visit Anna Bartkowski (UC 5th floor) or see many other options by visiting [this website](https://myusf.usfca.edu/title-IX)
# - Students may speak to someone confidentially, or report a sexual assault confidentially by contacting Counseling and Psychological Services at 415-422-6352
# - To find out more about reporting a sexual assault at USF, visit [USF’s Callisto website](https://usfca.callistocampus.org/)
# - For an off-campus resource, contact [San Francisco Women Against Rape](http://www.sfwar.org/about.html) 415-647-7273
