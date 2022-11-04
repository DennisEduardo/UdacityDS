import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

messages_dataset_file = './data/disaster_messages.csv'
categories_dataset_file = './data/disaster_categories.csv'
database_name = 'DisasterResponseData.db'

def load_data(messages_filepath, categories_filepath):
    """Load and merge messages and categories data, prepare target data for multi-output classifier
    
    Arguments:
        messages_filepath {str} -- CSV filepath of messages data.
        categories_filepath {str} -- CSV filepath of categories data.
    
    Returns:
        Dataframe -- Pandas Dataframe of merged data
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on = 'id')

    categories = df.categories.str.split(pat = ';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis = 1)

    df = pd.concat([df, categories], axis = 1)

    return df


def clean_data(df):
    """Feature Data cleaning.
    1. Drop duplicates and unnecessary data
    
    Arguments:
        df {Dateframe} -- Pandas Dataframe
    
    Returns:
        df -- Pandas Dataframe processed
    """
    # drop duplicates
    df = df.drop_duplicates()

    return df
    

def save_data(df, database_filename):
    """Save processed data to a SQLite file
    
    Arguments:
        df {Dataframe} -- Pandas Dataframe
        database_filename {str} -- storaged dataframe filename
    """
    engine = create_engine('sqlite:///./data/' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        save_data(df, database_filename)
        
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