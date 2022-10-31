# Disaster Response Pipeline Project
## Project Description:
This project aims to analyse the disaster data from Figure Eight and to build a model that classifies the messages into specific categories during disaster. The dataset contains 30000 twitter messages obtained from an earthquake in Haiti in 2010, an earthquake in Chile in 2010, floods in Pakistan in 2010, superstorm Sandy in the U.S.A. in 2012, and news articles spanning a large number of years and 100s of different disasters. The specific categories are 36 namely medical_help,child_alone,death, missing_people, floods etc. The project aims at developing a Natural Language processing tool that categorizes the messages. This project involves building an ETL pipeline that facilitates Extraction, Transformation and loading the disaster data, followed by a Machine Learning pipeline. A Web app is developed that gives the classification output for an input message. The classification output helps the emergency organizations to help the needy during a disaster. The emergency team can be informed immediately and the people affected during any disaster can be assisted. 

## Project Sections:
1. ETL Pipeline:
    - Pipeline to Extract, Transform and Load Data
    - Save it in SQLite Database
3. ML Pipeline:
    - Load data from SQLite
    - Text Data processing 
    - ML pipeline 
    - Using GridSearchCV to determine the final model for classification
5. Web Development:
    - Design using HTML
    - Build Web application using Flask

## File Structure
- app
  - template
  - master.html  # main page of web app
  - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py
  - classifier.pkl  # saved model 

- README.md

## Requirements:
This project requires Python 3 with pandas, numpy, sklearn, plotly, nltk, sqlalchemy, pickle installed.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Result:
The web app displays the Overview of the Training set and the distribution of the categories.
![image](https://user-images.githubusercontent.com/110763030/198878677-99a5f9f0-4254-49cb-aaf3-97cb23e58169.png)
![image](https://user-images.githubusercontent.com/110763030/198878714-ae078110-db2a-48ce-b504-769ab1b62ddf.png)

The web app classifies the messages sent during disaster into 36 categories mentioned above.
![image](https://user-images.githubusercontent.com/110763030/198879042-6bdcce35-b090-4ace-9eff-a9078f7e4d8f.png)
![image](https://user-images.githubusercontent.com/110763030/198879080-a2ed12f4-c1e3-4d48-bf64-087f5fe6b2f4.png)


## Acknowledgement:
Thank you Udacity for the project and Figure Eight for the dataset.

