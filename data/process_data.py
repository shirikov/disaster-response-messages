import sys
import pandas as pd
from sqlalchemy import create_engine

# Load data
def load_data(messages_filepath, categories_filepath):
    
    '''Load and merge data on messages and categories.'''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = messages.merge(categories, how='left', on='id')
    
    # Create columns for categories of messages
    categories = merged_df.categories.str.split(";", expand=True)
    
    # Extract category names for column names
    categories.columns = [(lambda x: x[:-2])(x) for x in categories.iloc[0]]
    
    # Set each value to be 0 if the last character of the string is zero, 
    # 1 otherwise
    for col in categories:
        categories[col] = 1 - (categories[col].str[-1] == '0').astype(int)
        
    # Sort categories alphabetically
    categories = categories.reindex(columns=sorted(categories.columns))

    # Drop the original categories variable, replace with new dummies
    merged_df = pd.concat([merged_df.drop(columns=['categories']), categories], 
                          sort=False, axis=1)

    return merged_df

def clean_data(df):
    
    '''Check and drop duplicates.'''
    
    print(str(df[df.duplicated()].shape[0]), ' duplicate rows')
    
    cleaned_df = df.drop_duplicates()
    
    print('All duplicates deleted')
    
    return(cleaned_df)

def save_data(df, data_filename):
    
    '''Save data to SQL.'''
    
    # Save to SQL
    engine = create_engine('sqlite:///' + data_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, data_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(data_filepath))
        save_data(df, data_filepath)
        
        print('Cleaned data saved to database')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

    




