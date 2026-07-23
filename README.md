# Deep-ML Solutions

---

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)

A from-scratch solutions journal for [Deep-ML.com](https://www.deep-ml.com) — the algorithms behind linear algebra, statistics, calculus, classic machine learning, ensemble methods, deep learning, and NLP, each implemented by hand and explained rather than imported from a library.

**100+ problems solved so far**, and growing as new challenges are attempted.

<br>

## What's in this repo

In this repo you'll find two things:

- **`Deep-ML.ipynb`** — the main notebook, every solved problem in the order it was tackled.
- **`visualizer/`** — a Streamlit app that explains and visualizes any ML concept on demand, described below.

Each problem in the notebook follows the same pattern: a markdown title cell, followed by a plain-Python implementation, and — where it adds value — a NumPy/vectorized version right next to it, with comments walking through the logic step by step. The goal isn't just a passing solution, it's understanding *why* it works.

<br>

## Topics covered

The 100+ problems span nine areas, roughly in order of how much ground they cover:

- **Linear Algebra** (23) — matrix inverse, eigenvalues, row echelon form, CSR/CSC sparse formats, orthogonal projection
- **Classification & Evaluation Metrics** (20) — precision/recall/F1, ROC & AUC, confusion matrix, Jaccard & Dice score, binary cross-entropy
- **Machine Learning Fundamentals** (14) — linear regression via normal equation & gradient descent, ridge/L2 regularization, feature scaling, early stopping
- **Probability & Statistics** (12) — Poisson/Binomial/Normal distributions, conditional probability, KL divergence
- **Deep Learning & Neural Networks** (10) — the ReLU/sigmoid/softmax family, a single neuron with backpropagation, batch iteration
- **Ensemble Learning & Model Validation** (8) — bagging, random forest, hard/soft voting, k-fold cross-validation
- **Classical ML Models** (4) — k-nearest neighbors, decision trees, SVM margins
- **Calculus** (3) — gradients, chain rule, product rule
- **NLP** (1) — a character-level tokenizer (stoi/itos/BOS)

This is a living breakdown, not a fixed list — open the notebook and search for any `***Title***` cell to jump straight to a solution, or use the app below to explore a concept before (or instead of) reading the code.

<br>

## The Deep-ML Explorer app

`visualizer/` is a companion Streamlit app for building intuition before diving into code. Type in any concept — say "Orthogonal Projection" or something not even in this repo, like "Batch Normalization" — and get:

- A worked example with real numbers
- A step-by-step breakdown of the math
- A matplotlib visualization of what's actually happening

Concepts already covered by this repo render instantly from a local library; anything else is generated on the fly by Gemini and cached for the session so it's never called twice for the same question.

**Run it locally:**

```bash
cd visualizer
pip install -r requirements.txt
streamlit run app.py
```

Optionally drop a `GEMINI_API_KEY` into a `.env` file in the project root (or paste it into the sidebar) to unlock explanations for concepts outside the pre-built library — the app works without one, just with a smaller set of concepts.

<br>

## Getting started with the notebook

```bash
git clone https://github.com/aboudrari/Deep-ML-Solutions.git
cd Deep-ML-Solutions

pip install numpy jupyter
jupyter notebook Deep-ML.ipynb
```

Every problem is self-contained — run any cell independently, no shared setup required between them.

<br>

## Author

**Abdallah Aboudrari** — AI Engineering Student @ Cyprus International University
[![GitHub](https://img.shields.io/badge/GitHub-aboudrari-181717?style=flat&logo=github)](https://github.com/aboudrari)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abdallah--aboudrari-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/abdallah-aboudrari)

---

> Solving these problems from scratch is part of a longer path toward mastering ML fundamentals. More problems — and more explorer concepts — get added as the journey continues.
