[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/5f7lMH9Y)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10587647&assignment_repo_type=AssignmentRepo)
# Assignment 3 - Language modelling and text generation using RNNs

## Contribution
- This assignment was made in contribution with fellow students from class, and with inspiration from in-class notebooks. In-code comments are made by me. The code in the *requirement_functions.py* script is made by teacher Ross.
- The data used in this assignment is *New York Times Comments* from articles published in Jan-May 2017 and Jan-April 2018. The data contains all the articles as csv files and the comments as csv files. There are over 2 million comments in total for over 9 thousand articles. According to the Kaggle author Aashita Kesarwani >"New York Times has a wide audience and plays a prominet role in shaping people's opinion and outlook on current affairs..." . This makes the data interesting to work with, since it is created by a range of different people, about real world matters. For more information about the data click [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

## Packages
- String
  - Is used to deal with strings. It is being used in the *requirement_functions.py*
- Os
  - Is used to navigate the operating system
- Pandas
  - Is used for data manipulation and structuring the data
- Numpy 
  - Is used to work with arrays and numerical data
- Argparse 
  - Is used to specify the path to the zip file as a command line argument
- Zipfile
  - Is used to extract the zip file
- Random
  - Is used to get a random sample
- Tensorflow
  - Is being used to import a range of functions, for preparing data, and to build the model.
- Sys
  - Is used to navigate the directory
- Joblib
  - Is used to save and load the models
  
## Assignment description 
For this assignemnt, you're going to create some scripts which will allow you to train a text generation model on some culturally significant data - comments on articles for *The New York Times*. You can find a link to the data [here](https://www.kaggle.com/datasets/aashita/nyt-comments).

You should create a collection of scripts which do the following:

- Train a model on the Comments section of the data
  - Save the trained model
- Load a saved model
  - Generate text from a user-suggested prompt

## Methods/What does the code do
- Script: *train.py*
  - Unzipping the zip file, joining all comments together in a list. Getting a sample of the list. Cleaning the corpus, tokenizing words, generating sequences of each comment and padding all comments so they are of equal length. Creating the model, training it and saving the model and tokenizer.
 - Script: *prompt.py*
  - Loading the model and tokenizer. Extracting the max sequence length from the model name, and generating text based on a one work user prompt.
 
## Usage 
To run this code follow these steps:
1. Clone the repository 
2. Get the zip file for the *New York Times Comments* from [here](https://www.kaggle.com/datasets/aashita/nyt-comments) and place it in the data folder
3. Run ```bash setup.sh``` in the command-line. This will install the requirements, and create a virtual environment. 
4. Run ```source ./assignment_3/bin/activate``` in the command-line. This will activate the virtual environment.
5. Run ```python3 src/train.py --zip_path data/archive.zip --sample_size 1000 --epochs 10``` in the command-line. This will run the code 
  - __OBS!__ *--zip_path* is your path to the zip file. *--sample_size* is how many comments you want. The default is all comments. *--epochs* is the amount of epochs the model should be trained on. Default is 10
  - Your model and tokenizer will be stored in the folder *out*
6. Run ```python3 src/prompt.py --filename out/rnn-model-seq_274.keras  --prompt Hello``` in the command-line. This will run the script that generates text.
  - __OBS!__ *--filename* is the path to the model you have created. *--prompt* is the prompt you want to generate text based of. The prompt should be one word
  - The existing model *rnn-model-seq_274.keras* is based on 1000 comments with 20 epochs and a final loss of 5.0161.
  - The text generation will appear in the command-line
