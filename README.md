# code_thesis_Floor_Halkes
This repository contains the code developed for my thesis project. Due to the large size of the data files, the dataset is not included in this repository.

## Getting Started
1. Clone the Repository
git clone <your-repo-url>
cd <your-repo-name>
2. Set Up the Environment with Poetry
Make sure you have Poetry installed.
poetry install
poetry shell

##  Data Preparation
Due to file size constraints, the required datasets are not tracked in the repository. 
To prepare your environment:
Create a data/ Folder


### Run Data Processing Scripts
From within the data_processing/ directory:
python clean_measurements.py
python clean_metadata.py
python create_helmond_nodes.py
These scripts clean and prepare the raw measurements, metadata, and generate the full network of Helmond.

## Create Training, Validation, and Test Data
Run the following script to generate subgraph datasets:
python create_training_data.py
This will create separate folders for training, validation, and test data.

## Model Training
Once your data is prepared, run:
python train.py
This script handles training, validation, and evaluation of the model.

## Notes
All dependencies are managed with Poetry and specified in pyproject.toml.

Ensure your directory structure matches what the scripts expect, especially under the data/ folder.
