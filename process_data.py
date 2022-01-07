import pandas as pd
import numpy as np

# Load and merge data
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')
df = messages.merge(categories, how='left', on='id')