# 3. Assignment 3 - Language modelling and text generation using RNNs
## 3.1 Assignment description
Written by Ross:

For this assignment, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for _The New York Times_. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments). You should create a collection of scripts which do the following:
- Train a model on the Comments section of the data and save the model. 
- Load the saved model and generate text from a user-suggested prompt.
## 3.2 Machine Specification and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. The scripts were created with Coder Python 1.73.1 and Python version 3.9.2.
### 3.2.1 Perquisites
To run the scripts, make sure to have Bash and Python3 installed on your device. The script has only been tested on Ucloud.
## 3.3 Contribution
This assignment was made in contribution with fellow students from class. The data used in this assignment is from Kaggle user [Aashita Kesarwani](https://www.kaggle.com/datasets/aashita/nyt-comments). 
### 3.3.1 Data
The data used in this assignment is _New York Times comments_ from articles published from Jan-May 2017 and Jan-April 2018. The dataset contains nine CSV files of articles and nine CSV files of comments. There are over 2 million comments in total for over 9 thousand articles. According to the Kaggle author Kesarwani _“New York Times has a wide audience and plays a prominent role in shaping people's opinion and outlook on current affairs.”_. This makes the data interesting to work with since it is created by a range of different people, about real-world matters. For more information about the data click [here](https://www.kaggle.com/datasets/aashita/nyt-comments).
## 3.4	Packages
- **String** is is being used in the ``requirement_functions.py`` to remove all punctuation form the comments.
- **Os** is used to navigate across operating systems.
- **Pandas** **(version 1.5.3)** is used to load the data, for data manipulation, and for structuring the data.
- **NumPy (version 1.24.2)** is used to work with arrays and numerical data.
- **Argparse** is used to specify the path to the zip file as a command line argument and for sampling.
- **Zipfile** is used to unpack the zip file.
- **Random** is used to get a random sample from a list.
- **Tensorflow (version 2.11.1)** is being used to import the following: _Sequential_ is used to create a sequential model. _Tokenizer_ is used to tokenize the text. _Pad_sequence_ is used to pad the text sequences, so they all have the same length. _Embedding_ is used to create word embeddings. _LSTM_ (long short-term memory) is used to remember long dependencies between words. _Dropout_ is used to remove weights during training to make it harder for the model to learn. _Dense_ is used to create the output layer for the model.
- **Sys** is used to navigate the directory.
- **Joblib** is used to save and load the models.
## 3.5	Repository Contents 
The repository contains the following folders and files:
- ***data*** is an empty folder where the zip file will be placed.
- ***models*** is the folder where the saved model will be stored
- ***src*** is the folder that contains the script for training the model, ``train.py``, and the script for testing the model, ``prompt.py``.
- ***utils*** is the folder that contains the script, ``requirement_functions.py``, which has functions used in the training script.
- ***README.md*** is the readme file.
- ***requirements.txt*** is a text file with version-controlled packages that are used in this repository.
- ***setup.sh*** is the file that creates a virtual environment, upgrades pip, and installs the packages from the requirements.txt.
## 3.6 Methods
### 3.6.1 Script: train.py
-	The script starts by initializing argparse to define the path to the zip file, create a sample size, and choose how many epochs to train the model on.
-	The script then unzips the zip file if it is not unzipped yet. 
-	After unzipping, a for loop is used to find the path to all CSV files that contain “_comments_” in their name, a data frame for each CSV file is then created, and the column “_commentBody_” is appended to a list.
-	A function for sampling is then created to reduce the amount of training data. I sampled 1000 comments out of the over 2 million available (due to memory issues).
-	The next function, ``cleaning_comments``, uses the function ``clean_text`` from the ``requirement_functions.py`` script to clean the text data. All text is turned to lowercase, and all punctuation is removed. 
-	The texts are then tokenized using TensorFlows _tokenizer_ function. 
-	The function ``input_sequence_function``, uses the ``get_sequence_of_tokens`` function from the ``requirement_functions.py`` script, which creates an n-gram for each comment. This helps the model learn longer distances.
-	The sequences are than padded to create equal lengths, using the function ``generate_padded_sequences`` from the ``requirement_function.py`` script. The padding is done by adding zeros to the start of the sequences. 
-	The recurring neural network (RNN) is then created with the function ``create_model`` from the ``requirement_function.py`` script. The model has the following architecture: It is a _sequential model_, with the first layer being the _embedding_ layer, and the next layer is the _LSTM_ layer with 100 neurons. The third layer is the _dropout_ layer, where 10% of the weights get dropped. The final layer is the _dense_ layer with _softmax_ _activation_, this is the layer that predicts the next word. Lastly, the model is compiled with the loss function _categorical_crossentropy_ and an _adam optimizer_.
-	The model is then trained with the comments. 
-	Lastly, the model is saved to the folder _out_ with the max sequence length as its name. The tokenizer is also saved. 
### 3.6.2 Script: prompt.py
-	This script starts by initializing two arg-parses. One is to get the path to the saved model (there might be more than one model), and the other is the user prompt.
-	The tokenizer and the saved model are then loaded. 
-	Next, the max sequence length is extracted from the model’s name using split. The function ``generate_text`` from the ``requirement_functions.py`` script is then used to generate text based on the user prompt. 
## 3.7 Discussion 
The RNN created in this script is used to generate text. This is done by having the model remember word dependencies in the text using _LSTM_. Thereby, the model should be able to predict the next word based on the previous words, by seeing which word fits the context the best. 

In theory, the output from the ``prompt.py`` script should be a 10-word sentence that makes sense. However, that was not the case since my model only was trained on 1,000 comments and 20 epochs, with a final loss of 5.01. This means that my model kept predicting the same words and got stuck in a loop. 

The corpus used in this assignment is an interesting and relevant dataset. The New York Times comments are created by people about real-world issues. Thereby, the model would have been trained on a corpus created by many different people, with different sentiments, and with different styles of writing. 
## 3.8 Usage
-	To run this code, follow these steps. 
-	Clone the repository.
-	Navigate to the correct directory.
-	Get the zip file for the New York Times Comments from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and place it in the data folder (you might need to rename the zip file). 
-	Run ``bash setup.sh`` in the command line. This will create a virtual environment and install the requirements.
-	Run ``source ./assignment_3/bin/activate`` in the command line. This will activate the virtual environment.
-	Run ``python3 src/train.py --zip_path data/archive.zip --sample_size 1000 --epochs 10`` in the command line. This will run the ``train.py`` script.
    - The argparse ``--zip_path`` takes a string as input and is the path to your zip file.
    - The argparse ``--sample_size`` takes an integer as input and has a default of 2,176,364 (all comments). Only include if you want to take a sample size of the data.
    - The argparse ``--epochs`` takes an integer as input and has a default of 10. Only change if you want to increase or decrease the number of epochs. 
-	Your model and tokenizer will be stored in the folder _out_
-	Run ``python3 src/prompt.py --filename out/rnn-model-seq_274.keras --prompt Hello`` in the command-line. This will run the script ``prompt.py``.
    - The argparse ``--filename`` takes a string as input and is the path to the model you created. 
    - The argparse ``--prompt`` is the prompt you want to generate text based on. The prompt should be one word
    - The existing model _rnn-model-seq_274.keras_ is based on 1000 comments with 20 epochs and a final loss of 5.0161.
    - The text generation will be printed to the command-line.
