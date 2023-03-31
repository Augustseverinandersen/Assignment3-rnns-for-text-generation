import string, os 
import argparse
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

from joblib import dump, load

import sys
sys.path.append("utils")
import requirement_functions as rf

from train import padded_sequences, tokenization

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    # add arguments // "--name" is what you feed it in the command line
    parser.add_argument("--filename", type=str)
    parser.add_argument("--prompt", type=str)
    # parse the arguments from command line
    args = parser.parse_args()
    return args




def required_variables():
    max_sequence_len = padded_sequences(input_sequence, total_words)
    tokenizer = tokenization(clean_data)
    return max_sequence_len, tokenizer

def saved_model(args):
    new_model = tf.keras.models.load_model(args.filename)
    new_model.summary()
    return new_model


def generate_text_function(args, model):
    # Get max sequence lenght from file name with split 
    tokenizer = load("out/tokenizer.joblib")
    filename = args.filename
    max_sequence_len = filename.split("_")[1].split(".")[0] # 1 means save everything to the right. # o mean everything to the left 
    print(rf.generate_text(tokenizer, args.prompt, 10, model, max_sequence_len)) # word you want, words to come after, model, make the sequence 24 in total.
    print(max_sequence_len)

def main_function():
    args = input_parse()
    #max_sequence_len, tokenizer = required_variables()
    new_model = saved_model(args)
    generate_text_function(args, new_model)


if __name__ == "__main__":
    main_function()