import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loads data from messages.csv and categories.csv files
    Input:
        messages_filepath: File path of messages data sets
        categories_filepath: File path of categories data
    Output:
        df: Merged dataset from messages and categories
    '''
    # read messages data sets
    messages = pd.read_csv(messages_filepath)
    # read categories data sets
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories on id column
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    clean input df
    Input:
        df: Merged dataset from messages and categories data sets, 
        returned df from load_data function
    Output:
        df: Cleaned dataset
    '''
    # split categories column into wide columns
    categories = df['categories'].str.split(';', expand=True)
    
    # take first row as header
    row = categories.iloc[0]
    # remove last 2 chars from columns
    category_colnames = list(map(lambda x : x[:-2], row))
    # rename the columns of categories datasets
    categories.columns = category_colnames
    # for each column in categories, take only 1 and 0 as values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from df
    df.drop(["categories"], axis = 1, inplace = True)
    # concat df and categories and return final df
    df = pd.concat([df, categories], axis=1)
    return df


def save_data(df, database_filename):
    '''
    Save final df into sqlite db
    Input:
        df: final clean dataset
        database_filename: database name
    Output: 
        A SQLite database
    '''
    # create SQLite engine
    engine =  create_engine('sqlite:///{}'.format(database_filename))
    # save final dataset to disaster_response table
    df.to_sql("Disaster_Response", engine, index=False)  


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