# NLP Final Project - ITI Applied AI Development

Final project of the NLP part of the ITI Applied AI Development track.

## Notebooks

### 1. `Arabic_News_NLP.ipynb` - Arabic news dataset (8,378 articles, 9 categories)

- Arabic preprocessing: diacritics and tatweel removal, letter normalization, Arabic stopwords
- Visualization: category distribution, article and summary lengths, top words per category
- N-grams (unigrams, bigrams, trigrams) and least repeated words
- POS tagging and NER with CAMeLBERT models
- Sentiment analysis with 3 pretrained models (CAMeLBERT-DA, CAMeLBERT-Mix, XLM-RoBERTa) and agreement between them
- Topic modeling with 3 models (LDA, LSA, NMF) compared by coherence score
- Category classification with TF-IDF + ML models (Logistic Regression, Naive Bayes, SVM)
- Fine tuning AraBERT for the same classification task
- Comparison of all the results

| Model | Accuracy | Macro F1 |
|---|---|---|
| AraBERT (fine tuned) | 0.823 | 0.795 |
| Linear SVM (TF-IDF, tuned) | 0.808 | 0.753 |
| Logistic Regression (TF-IDF, tuned) | 0.804 | 0.740 |
| Naive Bayes (TF-IDF, tuned) | 0.791 | 0.659 |

Best topic model: NMF (coherence 0.77).

### 2. `ML_Spam_Sentiment.ipynb` - ML models with TF-IDF

- SMS Spam Collection: Logistic Regression, Naive Bayes and SVM, then improved with word + character n-grams and class weights (best accuracy 99.0%)
- Book reviews sentiment (3 classes) with 5-fold cross validation, compared with the transformer results

## Data

All datasets are in the `data` folder:

- `summarizdataset.csv` - Arabic news articles with category and summary
- `spam.csv` - [SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- `book_reviews_sample.csv` - book reviews with ratings

## Run

The Arabic notebook needs a GPU (Google Colab T4 works). Main libraries: pandas, scikit-learn, nltk, gensim, transformers, torch, matplotlib, seaborn, arabic-reshaper, python-bidi.

## Author

Abdelrahman Hesham
