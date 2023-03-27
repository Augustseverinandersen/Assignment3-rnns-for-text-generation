import string, os 

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
from train import padded_sequences, tokenization


def required_variables():
    max_sequence_len = padded_sequences(input_sequence, total_words)
    tokenizer = tokenization(clean_data)
    return max_sequence_len, tokenizer

def saved_model():
    new_model = tf.keras.models.load_model('out')
    new_model.summary()
    return new_model


def generate_text_function(model, max_sequence_len):
    print(rf.generate_text("Hello", 10, model, max_sequence_len)) # word you want, words to come after, model, make the sequence 24 in total.

def main_function():
    max_sequence_len, tokenizer = required_variables()
    new_model = saved_model()
    generate_text_function(new_model, max_sequence_len)


if __name__ == "__main__":
    main_function()