# Disaster Response Pipeline Project

## Introduction

This project is part of The [Udacity] Data Scientist Nanodegree Program
The goal of this project is to apply the data engineering skills learned in the course to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.
The classified messages allow these messages to be sent to the appropriate disaster relief agency and delivered fast proper attendance.


## Software and Libraries

This project uses Python 3.7.2 and the following libraries:
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org)
* [nltk](https://www.nltk.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [sqlalchemy](https://www.sqlalchemy.org/)
* [dash](https://plot.ly/dash/)


## Data

The dataset is provided by [Figure Eight](https://www.figure-eight.com/dataset/combined-disaster-response-data/) is basically composed by:
* **disaster_categories.csv**: Categories of the messages
* **disaster_messages.csv**: Multilingual disaster response messages


## Folder Structure

<pre>
|-- disaster_response_pipeline_project
    |-- app                                <- Source code for this project
    |   |-- template                       <- Flask html templates
    |   |   |-- go.html
    |   |   |-- master.html
    |   |-- run.py                         <- Scripts to start
    |
    |-- data                               <- Raw and processed data
    |   |-- disaster_categories.csv        <- Raw categories data
    |   |-- disaster_messages.csv          <- Raw messages data
    |   |-- DisasterResponseData.db        <- Saved processed data
    |   |-- process_data.py                <- Scripts to process data
    |
    |-- model                              <- Trained models and ML pipeline
    |   |-- classifier.pkl                 <- Saved model
    |   |-- train_classifier.py            <- Scripts to train model
    |
    |-- ETL Pipeline Preparation.ipynb     <- Jupyter notebooks
    |-- ML Pipeline Preparation.ipynb      <- Jupyter notebooks
	|-- dash1.jpg                          <- Dashboard image to display in app web page
	|-- dash2.jpg                          <- Dashboard image to display in app web page
    |-- README.md
</pre>


## Running the code

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing and Acknowledgements

Thank you [Figure Eight](https://www.figure-eight.com/) for the datasets and more information about the licensing of the data can be find [here](https://www.figure-eight.com/datasets/).