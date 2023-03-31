# Data processing tools
import string, os 
import pandas as pd
import numpy as np
np.random.seed(42)
import random 

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

# Importing functions for folder utils
import sys
sys.path.append("utils")

import requirement_functions as rf




# Loading data 
def filepath():
    print("Loading data")
    data_dir = os.path.join("data", "news_data") # Loading data from folder data.
    return data_dir  



# Appending columns
def creating_list(file_path): 
    all_comments = [] # Creating empty list
    for filename in os.listdir(file_path): # Creating a list of file path to the comments.
        if 'Comments' in filename: # Only take files that have comments in their name.
            comment_df = pd.read_csv(file_path + "/" + filename) # Creating a dataframe for each file.
            all_comments.extend(list(comment_df["commentBody"].values)) # Taking column "commentsBody" and appending into the 
            #empyt list. Thereby creating a list of only comments. 
    print("Amount of comments: " + str(len(all_comments))) # Printing the length of the list
    return all_comments



# Sampling
def data_sampling(comments_list):
    print("Creating Sampel")
    thousand_comments = random.sample(comments_list, 10) # Taking 1000 random comments from the list.
    print("Sample size: " + str(len(thousand_comments))) # Printing sample length
    return thousand_comments



# Cleaning
def cleaning_comments(sample_list):
    print("Cleaning text")
    corpus = [rf.clean_text(x) for x in sample_list] # Using Ross' function to remove all punctations from the sampled list.
    return corpus


   
# Tokenizing
def tokenization(clean_data):
    print("Tokenizing")
    tokenizer = Tokenizer() # Placing tensor flow function for tokenizing wordsin a variable
    
    tokenizer.fit_on_texts(clean_data) # tokenizing the text, and gives every word an index. Creating a vocab.
    total_words = len(tokenizer.word_index) + 1 # how many total words are there. 
    #The reason for + 1 is to account for  = out of vocabulary token. if the tensorflow does not know the word. <unk> unknown word.
    return tokenizer, total_words  



# Sequence
def input_sequence_function(tokenizer, clean_data):
    print("Input sequence")
    # Each comment has multiple rows. 1-2, 1-2-3, 1-2-3-4 words (n-grams)
    inp_sequences = rf.get_sequence_of_tokens(tokenizer, clean_data) # Creating n-grams for each comment (that is tokenized)
    # with Ross' function "Get sequence_of_tokens"
    # Teaching the model to account for longer distances. 
    return inp_sequences



# Padding
def padded_sequences(input_sequence, total_words):
    print("Padding sequences")
    predictors, label, max_sequence_len = rf.generate_padded_sequences(input_sequence, total_words) # Using Ross' function
    # All inputs need to be same length. 
    # Adding zeros to the start of short sequences 
    # predictors = input vectors 
    # labels = words 
    print("Max sequence length: " + str(max_sequence_len)) # Printing the maximum sequence length
    return predictors, label, max_sequence_len



# Creating Model
def creating_model(sequnece_length, total_words):
    print("Creating model")
    model = rf.create_model(sequnece_length, total_words) # Using Ross' function to create a model.
    print(model.summary()) # Printing the model summary
    return model



# Training model
def training_model(model, predictors, label):
    print("Training model")
    # Training the model, and saving training data in a variable.
    history = model.fit(predictors, # Input vectors 
                        label, # Words
                        epochs=1, # How many runs should the model do
                        batch_size=128, # Bach size. Update weights after 128 comments
                        verbose=1) # Print status 
    return history



# Saving model
def saving_model(model, max_sequence_len):
    folder_path = os.path.join(f"out/rnn-model-seq_{max_sequence_len}.keras") # Defining out path
    tf.keras.models.save_model( # Using Tensor Flows function for saving models.
    model, folder_path, overwrite=True, save_format=None 
    ) # Model name, folder, Overwrite existing saves, save format = none 

def saving_tokenizer(tokenizer):
    from joblib import dump, load # Importing joblibs, dumb and load functions.
    dump(tokenizer, "out/tokenizer.joblib") # Saving tokenizer as a joblib, to be used in other script

def main_function(): # Running all functions with true paramenters.
    data_dir = filepath() # finding file path
    all_comments = creating_list(data_dir) # Creating dataframe and list of comments
    thousand_comments = data_sampling(all_comments) # samplining comments in new list
    corpus = cleaning_comments(thousand_comments) # removing punctuation from comments
    tokenizer, total_words = tokenization(corpus) # tokenizing words 
    inp_sequences = input_sequence_function(tokenizer, corpus) # generating sequence of each comment
    predictors, label, max_sequence_len = padded_sequences(inp_sequences, total_words) # padding sequence to create equal length
    model = creating_model(max_sequence_len, total_words) # creating model architecture 
    history = training_model(model, predictors, label) # training model
    saving_model(model, max_sequence_len) # saving model
    saving_tokenizer(tokenizer) # saving tokenizer
    print("done")



if __name__ == "__main__": # If called from terminal run main function
    main_function()


# 268 - comments 10 - epoch 100
# 284 - comments 1000 - epoch 5