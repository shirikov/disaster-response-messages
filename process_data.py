import pandas as pd
from sqlalchemy import create_engine

# Load and merge data
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')
df = messages.merge(categories, how='left', on='id')

# Create columns for categories of messages
categories = df.categories.str.split(";", expand=True)

# Extract category names for column names
categories.columns = [(lambda x: x[:-2])(x) for x in categories.iloc[0]]

# Set categories to 0 or 1
for column in categories:
    # Set each value to be 0 if the last character of the string is zero, 1 otherwise
    categories[column] = 1 - (categories[column].str[-1] == '0').astype(int)
    
# Sort categories alphabetically
categories = categories.reindex(columns=sorted(categories.columns))

# Drop the original categories variable, replace with new dummies
df = pd.concat([df.drop(columns=['categories']), categories], sort=False, axis=1)

# Check the number of duplicates, drop them
df[df.duplicated()].shape[0]
df = df.drop_duplicates()

# Save to SQL
engine = create_engine('sqlite:///disaster_messages.db')
df.to_sql('disaster_messages', engine, index=False, if_exists='replace')