import sys
import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

def load_data(database_filepath):
    '''
    Load data sets from disaster_reponse database
    Input:
        database_filepath: File path of SQLite database
    Output:
        X: message feature
        Y: categories target features
        category_names: all target categirues names as array
    '''
    print('loading disaster responses datasets...............')
    # create SQLite database engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # read data from disaster_response table
    df = pd.read_sql("SELECT * from Disaster_Response", engine)
    # fetch X feature values and Y target values
    X = df['message']
    Y = df.iloc[:,4:]
    # calculate category names (target labels)
    category_names = list(df.columns[4:])
    print(X.head())
    print(Y.head())
    print(category_names)

    return X, Y, category_names

def tokenize(text):
    '''
    Tokenize text
    Input:
        text: message text
    Output:
        lemmatized_tokens: Tokenized and lemmatized text
    '''
    reg_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(reg_url, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) == 0:
                return False
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
                
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
       
def build_model():
    """
    Build model using pipeline and GridSearchCV.
    this function uses sklearn Pipeline and GridSearchCV to construct model.
    
    Input: None
    Output: instance of GridSearchCV
    """
    print('Building Disaster Response model...')
    # create instance of Pipeline
    # Use feature Union to create an extra feature
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    # parameters for fine tuning
    parameters = {}
    parameters["clf__estimator__n_estimators"] = [100]
    # create instance of GridSearchCV by passing pipeline and param instances
    model = GridSearchCV(pipeline, parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluate model using test datasets. 
    Prints accuracy, precision and recall scores
    Input: 
        model: Model instaqnce to be evaluated
        X_test: Test data samples
        Y_test: True lables for Test data
        category_names: Target labels
    Output:
        Print accuracy, precision and recall score for each category
    '''
    print('Evaluating Diaster Response model...')
    # predict using model
    y_pred =  model.predict(X_test)
    # display results
    display_results(Y_test, y_pred, category_names)
    

        
def display_results(Y_test, y_pred, category_names):
    # iterate all col and print scores for each column
    for i, col in enumerate(category_names):
        # print('column name:: {}'.format(col))
        print("Category:", col, "\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(col, accuracy_score(Y_test.iloc[:, i].values, y_pred[:, i])))
        # calculate accuracy
        # accuracy = accuracy_score(Y_test[i], y_pred[i])
        # calculate precision
        # precision = precision_score(Y_test[i], y_pred[i])
        # calculate recall score
        # recall = recall_score(Y_test[i], y_pred[i])
        # print("\tAccuracy:: {0:.2f} \tPrecision:: {0:.2f} \tRecall:: {0:.2f}\n".format(accuracy, precision, recall))
        #print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        
        
def save_model(model, model_filepath):
    '''
    This function saves model as a pickle file 
    Input: 
        model: Model to be saved
        model_filepath: path of the stored pickle file
    Output:
        A pickle file of stored model
    '''
    print("Saving Disaster Response Model...")
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
