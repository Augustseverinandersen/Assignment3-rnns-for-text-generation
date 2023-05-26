# Data processing tools
import string, os 
import pandas as pd
import numpy as np 
np.random.seed(42)
import argparse # Command line arguments
import zipfile # Zip file manipulation
import random # Random sampling

# keras module for building LSTM model
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 

from tensorflow.keras.preprocessing.text import Tokenizer


# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore warnings from libraries. 

# Importing functions for folder utils
import sys
sys.path.append("utils")

import requirement_functions as rf # Functions created by Ross




# Defining a function for the user to input a filepath
def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, help = "Path to the zip file")
    parser.add_argument("--sample_size", type = int, default = 2176364, help = "Specify amount of comments you want to use")
    parser.add_argument("--epochs", type = int, default = 10, help = "How many epochs")
    args = parser.parse_args()

    return args


def unzip(args):
    folder_path = os.path.join("data", "news_data") # Defining the folder_path to the data.
    if not os.path.exists(folder_path): # If the folder path does not exist, unzip the folder, if it exists do nothing 
        print("Unzipping file")
        os.mkdir(folder_path) # Creating the folder news_data
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = folder_path # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination) # Unzipping
    print("The files are unzipped")
    return folder_path


# Appending columns
def creating_list(file_path): 
    all_comments = [] # Creating empty list
    for filename in os.listdir(file_path): # Creating a list of file paths to the comments.
        if 'Comments' in filename: # Only take files that have comments in their name. Some have article in the name.
            comment_df = pd.read_csv(file_path + "/" + filename) # Creating a dataframe for each file.
            all_comments.extend(list(comment_df["commentBody"].values)) # Taking column "commentsBody" and appending into the 
            # empty list. Thereby creating a list of only comments. 
    print("Amount of comments: " + str(len(all_comments))) # Printing the length of the list
    return all_comments


# Sampling
def data_sampling(comments_list, args):
    print("Creating Sampel")
    thousand_comments = random.sample(comments_list, args.sample_size) # Taking a user specified amount of random comments from the list.
    print("Sample size: " + str(len(thousand_comments))) # Printing sample length
    return thousand_comments


# Cleaning
def cleaning_comments(sample_list):
    print("Cleaning text")
    corpus = [rf.clean_text(x) for x in sample_list] # Using function from utils folder to remove all punctuation from the sampelled list.
    return corpus


# Tokenizing
def tokenization(clean_data):
    print("Tokenizing")
    tokenizer = Tokenizer() # Placing tensor flow function for tokenizing words in a variable
    
    tokenizer.fit_on_texts(clean_data) # tokenizing the text, and gives every word an index. Creating a vocab.
    total_words = len(tokenizer.word_index) + 1 # how many total words are there. 
    # The reason for + 1 is to account for  = out of vocabulary token. if the tensorflow does not know the word. <unk> unknown word.
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
    # Adding zeros to the start of short sequences to make them same length
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
def training_model(model, predictors, label, args):
    print("Training model")
    # Training the model, and saving training data in a variable.
    history = model.fit(predictors, # Input vectors 
                        label, # Words
                        epochs= args.epochs, # How many runs should the model do
                        batch_size=32, # Batch size. Update weights after 32 comments
                        verbose=1) # Print status 
    return history


# Saving model
def saving_model(model, max_sequence_len):
    print("Saving Model")
    folder_path = os.path.join(f"out/rnn-model-seq_{max_sequence_len}.keras") # Defining out path 
    # The reason for f string, is because the max_sequence_len is being used in the prompt.py script
    tf.keras.models.save_model( # Using Tensor Flows function for saving models.
    model, folder_path, overwrite=True, save_format=None 
    ) # Model name, folder, Overwrite existing saves, save format = none 

def saving_tokenizer(tokenizer):
    print("Saving Tokenizer")
    from joblib import dump, load # Importing joblibs, dumb and load functions.
    dump(tokenizer, "out/tokenizer.joblib") # Saving tokenizer as a joblib, to be used in other script

def main_function(): # Running all functions with true paramenters.
    args = input_parse() # Command line arguments
    folder_path = unzip(args) # Unzipping zip file function
    all_comments = creating_list(folder_path) # Creating dataframe and list of comments
    thousand_comments = data_sampling(all_comments, args) # samplining comments in new list
    corpus = cleaning_comments(thousand_comments) # removing punctuation from comments
    tokenizer, total_words = tokenization(corpus) # tokenizing words 
    inp_sequences = input_sequence_function(tokenizer, corpus) # generating sequence of each comment
    predictors, label, max_sequence_len = padded_sequences(inp_sequences, total_words) # padding sequence to create equal length
    model = creating_model(max_sequence_len, total_words) # creating model architecture 
    history = training_model(model, predictors, label, args) # training model
    saving_model(model, max_sequence_len) # saving model
    saving_tokenizer(tokenizer) # saving tokenizer
    print("done")


if __name__ == "__main__": # If called from terminal run main function
    main_function()
