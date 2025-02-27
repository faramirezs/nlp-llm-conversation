#!/bin/bash

#check which models are available by runing:
ollama list

#Grep the ouput of previous command, ennumerate the list from the previous step and add a final option with the option of all.

# Prompt the user for which
echo "Enter the model name:"
read model_name

# Update the model_config.yaml file with the provided model name
sed -i '' "s/^model: name: \".*\"/model: name: \"$model_name\"/" model_config.yaml

# Run the Python script
python3 open_source_examples/model_playground.py data/groups/thisiscere/messages_thisiscere.csv
