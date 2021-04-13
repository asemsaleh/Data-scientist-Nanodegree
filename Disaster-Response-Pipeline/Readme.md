# Disaster Response Pipeline Project
In this project, the task is to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set contains real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories.
### Install
This project requires Python 3.x and the following Python libraries installed:
NumPy, Pandas, Matplotlib, Json
Plotly, Nltk, Flask, Sklearn
Sqlalchemy, Sys, Re, Pickle
### Code and data
* process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
* train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
* ETL Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py automates this notebook.
* ML Pipeline Preparation.ipynb: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which model to use. train_classifier.py automates the model fitting process contained in this notebook.
* disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
* Templates folder: This folder contains all of the files necessary to run and render the web app.

### Instructions
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database:
* python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves:
* python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app:
* python run.py
Go to http://0.0.0.0:3001/

