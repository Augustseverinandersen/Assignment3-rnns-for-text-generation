
#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_3

# Activate virtual enviroment 
source ./assignment_3/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the code 
python3 src/train.py 
python3 src/prompt.py --filename out/rnn-model-seq_268.keras --prompt Hello

# Deactivate the virtual environment.
deactivate