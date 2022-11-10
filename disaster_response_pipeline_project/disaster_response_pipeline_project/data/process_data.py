import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

messages_dataset_file = './data/disaster_messages.csv'
categories_dataset_file = './data/disaster_categories.csv'
database_name = 'DisasterResponseData.db'

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe
    
    Inputs:
    messages_filepath    filepath to messages csv file
    categories_filepath  filepath to categories csv file
    
    Returns:
    df      dataframe merging categories and messages
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on = 'id')

    categories = df.categories.str.split(pat = ';', expand = True)
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x:x[:-2]).to_list()
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis = 1)

    df = pd.concat([df, categories], axis = 1)

    return df


def clean_data(df):
    '''
    clean_data
    Data cleaning to drop duplicates and unnecessary data
    
    Inputs
    df      pandas dataframe with merge categories and messages
    
    Returns:
    df      dataframe cleaned data
    '''

    # drop duplicates
    df = df.drop_duplicates()

    return df
    

def save_data(df, database_filename):
    '''
    save_data
    Save processed data to a SQLite file
    
    Inputs:
    df                 pandas dataframe
    database_filename  filename to storaged data
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists = 'replace', index=False)
    return


def dashboads_update(df):
    '''
    dashboads_update
    Generate and save graphs to web app page
    
    Inputs:
    df                 pandas dataframe
    '''

    dfg = df.groupby('genre')['message'].count().reset_index(name='qty')
    fig, ax = plt.subplots(figsize=(17,11), dpi=300)
    ax.bar(dfg.genre, dfg.qty, color='green')
    ax.set_xlabel('\n Genre', fontsize=16)
    ax.set_title('Distribution of Messages Genre \n', fontsize=23)
    plt.savefig('../dash1.jpg', format='jpg', dpi=300)

    dfg2 = df.sum().drop(['id','message','genre']).sort_values(ascending=False)[:9].reset_index()
    dfg2.columns = ['response','qty']
    fig, ax = plt.subplots(figsize=(17,11), dpi=300)
    ax.bar(dfg2.response, dfg2.qty, color='green')
    ax.set_xlabel('\n Response', fontsize=16)
    ax.set_title('Top 10 response \n', fontsize=23)
    plt.savefig('dash2.jpg', format='jpg', dpi=300)
    return


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
        
        print('Update Dashboards!')
        dashboads_update(df)
        
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