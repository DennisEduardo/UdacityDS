import sys
import re
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import pickle

import warnings
warnings.simplefilter('ignore')

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y,list(Y.columns.values)


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = text.lower()

    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tok = []
    for tok in tokens:
        clean_tok.append(lemmatizer.lemmatize(tok).strip())

    return clean_tok


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0)
        , 'clf__estimator__n_estimators': [50, 100]
        , 'clf__estimator__min_samples_split': [2, 3, 5]
    }

    model = GridSearchCV(pipeline, param_grid = parameters, cv=3, scoring='accuracy', n_jobs = -1)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    target_names = category_names

    for i in range(len(target_names)):
        print('Target category {}: {} '.format(i, target_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Confusion Matrix: \n {} \n\n'.format(confusion_matrix(y_pred[:, i], Y_test.iloc[:, i])))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))

    print(classification_report(Y_test.iloc[:, 1:].values, np.array([x[1:] for x in y_pred]), target_names = target_names[1:]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()