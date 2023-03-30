
#!/usr/bin/env bash

# Create virtual enviroment 
#python3 -m venv assignment_3

# Activate virtual enviroment 
#source ./assignment_3/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_md

# Run the code 
#python3 src/train.py --filename ./out/
#python3 src/neural_network_classifier.py 

# Deactivate the virtual environment.
#deactivate