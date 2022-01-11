import pandas as pd
from sqlalchemy import create_engine

# Load and merge data
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')
df = messages.merge(categories, how='left', on='id')

# Create columns for categories of messages
categories = df.categories.str.split(";", expand=True)

# Extract category names for column names
category_colnames = [(lambda x: x[:-2])(x) for x in categories.iloc[0]]
categories.columns = category_colnames

# Set categories to 0 or 1
for column in categories:
    # Set each value to be the last character of the string
    categories[column] = pd.to_numeric(categories[column].str[-1])

# Drop the original categories variable, replace with new dummies
df = df.drop(columns=['categories'])
df = pd.concat([df, categories], sort=False, axis=1)

# Check the number of duplicates, drop them
df[df.duplicated()].shape[0]
df = df.drop_duplicates()

# Save to SQL
engine = create_engine('sqlite:///disaster_messages.db')
df.to_sql('disaster_messages', engine, index=False)