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


# Used to clean the corpus in train.py
def clean_text(txt): # return vocab if it is not part of string.punctuation 
    # string.punctuation is a python model. ( a list of all string characters that er punctuations /%&¤#";:_-.,*")
    txt = "".join(v for v in txt if v not in string.punctuation).lower() # Making lower case 
    txt = txt.encode("utf8").decode("ascii",'ignore') # encoding utf8
    return txt 


# Used in the input_sequence_function function in train.py
def get_sequence_of_tokens(tokenizer, corpus):
    # convert data to sequence of tokens 
    input_sequences = [] # Creating empty list
    for line in corpus: # every line in the corpus 
        token_list = tokenizer.texts_to_sequences([line])[0] # create list of tokens 
        for i in range(1, len(token_list)): # order dem sequentialy
            n_gram_sequence = token_list[:i+1] # Creates an n_gram for each token_list
            input_sequences.append(n_gram_sequence) # Appending to input list
    return input_sequences


# Used in padded_sequence function in train.py
def generate_padded_sequences(input_sequences, total_words):
    # Get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # Make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1] # Getting last element to be used as label
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

# Used in the train.py script
def create_model(max_sequence_len, total_words): # model initilisation 
    input_len = max_sequence_len - 1
    model = Sequential() # sequential model
    # Add Input Embedding Layer
    model.add(Embedding(total_words, #
                        10, 
                        input_length=input_len))
    # Add Hidden Layer 1 - LSTM Layer
    model.add(LSTM(100)) # long short term memory
    model.add(Dropout(0.1)) # drop out layer, during training everytime you make an iteration 10% of the weights should be removed. 
    # so every iteration is only 90 %. Making things a bit more diffiuclt for the model 
    # Add Output Layer
    model.add(Dense(total_words, 
                    activation='softmax')) # Softmax prediction.
    model.compile(loss='categorical_crossentropy', 
                    optimizer='adam')
    
    return model

# Used in the prompt.py script
def generate_text(tokenizer, seed_text, next_words, model, max_sequence_len): # seed_text = prompt.
    for _ in range(next_words): # for how ever many in next_word.
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # get vocab 
        token_list = pad_sequences([token_list],  # pad it (zeros)
                                    maxlen=int(max_sequence_len)-1, 
                                    padding='pre')
        predicted = np.argmax(model.predict(token_list), # predict the next words with higest score.
                                            axis=1)
        
        output_word = ""
        for word,index in tokenizer.word_index.items(): # appending words together. 
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
