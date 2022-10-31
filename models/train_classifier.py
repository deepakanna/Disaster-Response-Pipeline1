import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report
import pickle
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    load_data:
    Load the data from SQLite database
    Split the data into feature and target variables
        
    Input:
    The filepath of the database file
    
    Returns:
    The feature,target variables and the column names of the target variable
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    Y = df.drop(['message', 'id', 'genre', 'original'], axis=1)
    X = df['message']
    category_names=Y.columns
    return X,Y,category_names


def tokenize(text):
    '''
    tokenize:
    Tokenization function to process the text data
        
    Input:
    Text data
    
    Returns:
    Clean processed data
    '''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', " ", text)
    tokens = word_tokenize(text)
    words = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in words:
        tok = lemmatizer.lemmatize(w).lower().strip()
        clean_tokens.append(tok)
    return clean_tokens


def build_model():
    '''
    build_model:
    Build a machine learning pipeline and uses GridSearch to determines the best parameters 
        
    Returns
    Grid Search object
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    #parameters = {'clf__estimator__criterion': ['gini', 'entropy']}
    parameters={'clf__estimator__min_samples_split':[2]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model:
    The built model is tested for the test data
        
    Input:
    The built model, the test feature, test target variable and the column names of the target variable
    The classification report for all the categories are displayed  
    
    '''
    #model.fit(X_train, y_train)
    #model.predict(X_test)
    #print(model.best_params_)
    #print(model.best_estimator_)
    #best = model.best_estimator_
    y_pred = model.predict(X_test)
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i], target_names=category_names))


def save_model(model, model_filepath):
    '''
    save_model:
    The built model is exported as a pickle file
        
    Input:
    The model and the filepath of the pickle file
    
    '''
    with open(model_filepath,'wb')as f:
        pickle.dump(model,f)


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