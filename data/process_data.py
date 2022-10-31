import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe
    Input:
    message_filepath file path to messages csv file
    categories_filepath filepath to categories csv file
    returns:
    df dataframe merging categories and messages
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages.head()
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #categories.head()
    # merge datasets
    df = messages.merge(categories, how='inner', on=['id'])
    #df.head()
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';', expand=True).values)
    categories.head()
    row = categories.iloc[1]
    rw = [r[:-2] for r in row]

    # rename the columns of `categories`
    categories.columns = rw
    #categories.head()
    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:
        categories[column].astype(str)
        categories[column] = categories[column].str.split('-', expand=True)[1]
    categories = categories.astype(int)
    #categories.dtypes
    # drop the original categories column from `df`

    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    return df




def clean_data(df):
    '''
    clean_data
    Check for duplicates in the dataframe 
    Remove the duplicates using drop method
    Input:
    DataFrame df
    Returns:
    Dataframe with no duplicates
    '''
    # check number of duplicates
    df[df.duplicated()]
    # drop duplicates
    df = df.drop_duplicates()
    # check number of duplicates
    return df



def save_data(df, database_filename):
    '''
    save_data
    save data as an sql file
    Input:
    DataFrame df
    Returns:
    Dataframe is saved as SQLite database
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
