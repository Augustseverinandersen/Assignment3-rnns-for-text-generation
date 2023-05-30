import string, os 
import argparse # Command line arguments
# keras module for building LSTM 
import tensorflow as tf
tf.random.set_seed(42)
import tensorflow.keras.utils as ku 


# surpress warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore warnings from libraries. 

from joblib import dump, load # Importing joblis to Get the saved models

import sys
sys.path.append("utils") # Find in utils folder

import requirement_functions as rf # Importing functions from utils folder

from train import padded_sequences, tokenization # Importing from script train.py

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--filename", type=str, help= "Your path to the saved model")
    parser.add_argument("--prompt", type=str, help = "One word")
    # parse the arguments from command line
    args = parser.parse_args()
    return args


def saved_model(args):
    tokenizer = load("out/tokenizer.joblib") # Loading the saved tokenizer
    new_model = tf.keras.models.load_model(args.filename) # Loading in the saved model
    new_model.summary() # Getting the model summary
    return tokenizer, new_model


def generate_text_function(args, model, tokenizer):
    # Get max sequence lenght from file name with split 
    filename = args.filename # getting the filename of the model.
    max_sequence_len = filename.split("_")[1].split(".")[0] # 1 means save everything to the right. # o mean everything to the left 
    # The max sequence length is saved in the model name. I am extracting it here to be used below
    print(rf.generate_text(tokenizer, args.prompt, 10, model, max_sequence_len)) # preprocessing, word you define, words to come after, model, max sequence length.


def main_function():
    args = input_parse() # Command line arguments
    tokenizer, new_model = saved_model(args) # Load the model
    generate_text_function(args, new_model, tokenizer) # Generate text from your prompt


if __name__ == "__main__": # If script is called from command line run the main function
    main_function()