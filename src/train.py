# data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)

# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore warnings from libraries. 

import sys
sys.path.append(".")
import utils.requirement_functions as rf


# Loading data 
print("Loading data")
data_dir = os.path.join("..","data", "news_data")


# Appending columns 
all_comments = []
for filename in os.listdir(data_dir):
    if 'Comments' in filename:
        comment_df = pd.read_csv(data_dir + "/" + filename) # joining data_dir / filename. ( Creating dataframe)
        all_comments.extend(list(comment_df["comments"].values)) # Creating a list of only headlines. 

# Cleaning up
all_headlines = [h for h in all_headlines if h != "Unknown"] # keep the headlines if they are not "unknown"
len(all_headlines)











































