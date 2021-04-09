import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def convert_categories_to_num(categories):
    '''
    INPUT: categories dataframe
    OUTPUT: converted data to binary
    This function convert each category to binary 0 or 1
    '''
    categories = categories['categories'].str.split(';',expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [ x.split("-")[0] for x in row]
    for col in categories:

        categories[col] = categories[col].map(
            lambda x: 1 if int(x.split("-")[1]) > 0 else 0 )
    return categories
        
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = convert_categories_to_num(pd.read_csv(categories_filepath))
    df = pd.merge(messages, categories, on='id', how='outer')
    return df


def clean_data(df):
    categories = df['categories'].str.split(";", expand=True)
    ow = categories.iloc[0]
    category_colnames = list(map(lambda x: x.split("-")[0], categories.iloc[0].values.tolist()))
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].apply(pd.to_numeric)
    del df['categories']
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('messages', engine, index=False,if_exists = 'replace')  


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
