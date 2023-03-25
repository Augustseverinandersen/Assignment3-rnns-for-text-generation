# data processing tools
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

#import sys
#sys.path.append("..")
#import utils.requirement_functions as rf






def clean_text(txt): # return vocab if it is not part of string.punctuation 
    # string.punctuation is a python model. ( a list of all string characters that er punctuations /%&¤#";:_-.,*")
    txt = "".join(v for v in txt if v not in string.punctuation).lower() # Making lower case 
    txt = txt.encode("utf8").decode("ascii",'ignore') # encoding utf8
    return txt 

def get_sequence_of_tokens(tokenizer, corpus):
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus: # every head 
        token_list = tokenizer.texts_to_sequences([line])[0] # list of tokens 
        for i in range(1, len(token_list)): # order dem sequentialy
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences

def generate_padded_sequences(input_sequences):
    # get the length of the longest sequence
    max_sequence_len = max([len(x) for x in input_sequences])
    # make every sequence the length of the longest on
    input_sequences = np.array(pad_sequences(input_sequences, 
                                            maxlen=max_sequence_len, 
                                            padding='pre'))

    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, 
                            num_classes=total_words)
    return predictors, label, max_sequence_len

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

def generate_text(seed_text, next_words, model, max_sequence_len): # seed_text = prompt.
    for _ in range(next_words): # for how ever many in next_word.
        token_list = tokenizer.texts_to_sequences([seed_text])[0] # get vocab 
        token_list = pad_sequences([token_list],  # pad it (zeros)
                                    maxlen=max_sequence_len-1, 
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





  # Loading data 
def filepath():
    print("Loading data")
    data_dir = os.path.join(".","data", "news_data")
    return data_dir  







    # Appending columns
def creating_list(file_path): 
    all_comments = []
    for filename in os.listdir(file_path):
        if 'Comments' in filename:
            comment_df = pd.read_csv(file_path + "/" + filename) # joining data_dir / filename. ( Creating dataframe)
            all_comments.extend(list(comment_df["commentBody"].values)) # Creating a list of only comments. 
    print("Amount of comments: " + str(len(all_comments)))
    return all_comments








def data_sampling(comments_list):
    thousand_comments = random.sample(comments_list, 1000)
    print("Sample size: " + str(len(thousand_comments)))
    return thousand_comments





def cleaning_comments(sample_list):
    print("Cleaning text")
    corpus = [clean_text(x) for x in sample_list]
    
    return corpus

   



def tokenization(clean_data):
    print("Tokenizing")
    tokenizer = Tokenizer()
    ## tokenization
    tokenizer.fit_on_texts(clean_data) # tokenizing the text, and gives every word an index. Creating a vocab.
    total_words = len(tokenizer.word_index) + 1 # how many total words are there. The reason for + 1 is to account for  = out of vocabulary token. if the tensorflow does not know the word. <unk> unknown word.
    
    return tokenizer, total_words  






def input_sequence_function(tokenizer, clean_data):
    print("Input sequence")
    inp_sequences = get_sequence_of_tokens(tokenizer, clean_data)
    # Each document has multiple rows. 1-2, 1-2-3, 1-2-3-4 words (n-grams)
    # Teaching the model to account to longer distances. 
    return inp_sequences







def padded_sequences(input_sequence):
    print("Padding sequences")
    predictors, label, max_sequence_len = generate_padded_sequences(input_sequence) 
    # All inputs need to be same lenght. 
    # adding zeros to the start of shorted sequences 
    # predictors = input vectors 
    # labels = words 
    print("Max sequence length: " + max_sequence_len)
    return predictors, label, max_sequence_len








def create_model(sequnece_length, total_words):
    print("Creating model")
    model = create_model(sequnece_length, total_words)
    print(model.summary())
    return model








def training_model(model):
    print("Training model")
    history = model.fit(predictors, 
                        label, 
                        epochs=1, # prev. 100
                        batch_size=128, # Updates weights after 128 
                        verbose=1)
    return history

# In notebooks, a models history is saved. So if the model has run one time with 100 epoch and you start it again it will run for 200 intotal.
# You either need to create the model again ( Above chunck) or use tensor flow functiion clear history.









def main_function():
    data_dir = filepath()
    all_comments = creating_list(data_dir)
    thousand_comments = data_sampling(all_comments)
    corpus = cleaning_comments(thousand_comments)
    tokenizer, total_words = tokenization(corpus)
    inp_sequences = input_sequence_function(tokenizer, corpus)
    predictors, label, max_sequence_len = padded_sequences(inp_sequences)
    model = create_model(max_sequence_len, total_words)
    history = training_model(model)



if __name__ == "__main__":
    main_function()

