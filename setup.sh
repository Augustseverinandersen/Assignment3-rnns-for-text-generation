
#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_3

# Activate virtual enviroment 
source ./assignment_3/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Deactivate the virtual environment.
deactivate