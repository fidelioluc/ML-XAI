# XAI for Document Classification with Local LLMs

## Overview 

This project explores how different text vectorization methods impact document classification and interpretability. We compare TF-IDF, Doc2Vec, and SBERT applied to LLM-generated summaries across multiple classifiers. 
Using LIME and a local LLM (Gemini 3B), we evaluate how well model decisions align with human-readable explanations. Few-shot and zero-shot scenarios are also explored using the AG News dataset.

## Datasets

- **Primary**: [20 Newsgroups](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)  
- **Few-shot/Zero-shot**: [AG News](https://www.kaggle.com/amananandrai/ag-news-classification-dataset)

## Results 
