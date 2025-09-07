NLP Text Classification Pipeline (Sentiment Analysis)
This repository contains a Jupyter Notebook implementing an NLP preprocessing pipeline and a text classification model for sentiment analysis using the IMDB reviews dataset.

Dataset
The dataset used is the IMDB Reviews Dataset.

Source: Kaggle - IMDB Dataset of 50K Movie Reviews
The dataset contains 50,000 movie reviews, labeled as either positive or negative sentiment.

Preprocessing Steps
The following preprocessing steps were applied to the raw text data:

Convert to lowercase: All text was converted to lowercase.
Remove URLs: URLs starting with http, https, or www were removed.
Remove punctuation: Punctuation marks were removed.
Remove numbers: Numerical digits were removed.
Remove extra whitespace: Multiple spaces were replaced with single spaces, and leading/trailing spaces were removed.
Tokenization: The text was split into individual words (tokens).
Remove stopwords: Common English stopwords (e.g., 'the', 'a', 'is') were removed.
Lemmatization: Words were reduced to their base or dictionary form using WordNetLemmatizer.
Remove single characters: Tokens with a length of 1 were removed.
Feature Engineering
TF-IDF Vectorization was used to transform the cleaned text into numerical features.

TfidfVectorizer was initialized with ngram_range=(1, 2), min_df=2, and max_features=10000.
The vectorizer was fitted on the training data and used to transform both the training and testing data.
Model Training
A Logistic Regression model was trained on the TF-IDF vectorized training data.

The model was initialized with max_iter=1000 and random_state=42.
Hyperparameter tuning was performed using GridSearchCV with the following parameters:
C: [0.1, 1, 10]
penalty: ['l2']
The best parameters found were {'C': 1, 'penalty': 'l2'}.
The best cross-validation accuracy score was approximately 0.89.
