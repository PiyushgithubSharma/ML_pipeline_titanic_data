# Titanic Survival prediction- Machine Learning Pipeline
A production ready Ml pipleline that predict whether a passanger survived the titanic disaster based on passenger information this project demostrate end to end ml workflow including data loading,feature engineering,model training,model evaluation 

# Proble statement
Given Passenger information,build a ML model that predicts wheter the passenger survived the titanic disaster

## target variable
    0-> Did not survive
    1-> Survive


## data Feature 
    In this we data which have many features but we only take some important data feature for this model like 
    
    Features                    information

    survived,           passenger survived or not
    pclass,             passenger class
    sex,                gender of passenger
    age,                age of passenger
    sibsp,              number of sibling of passenger 
    parch,              Number of parents/Children
    fare,               ticket Fare
    class               class of passenger's ticket

## ML Pipeline

    Titanic_data_eda -> Data Loading -> feature_eng -> model_training  -> model_evaluation -> model_prediction


##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Exploratory Data Analysis

The EDA notebook includes:

- Missing Value Analysis
- Data Distribution
- Correlation Analysis
- Feature Relationships
- Data Visualization


## Feature Engineering
    feature engineering includes:
        -Miising value Imputation
        -Categorical Encoding
        -Feature selection
        -Feature scaling
        - data transformation


##  model training

the model tains a classification model using
-Random forest Classifier

the trained model can also be experimented with:
Logistic Regression
Decision tree


## model Evaluation metrics

-Accuracy Score
-f1-score
-confusion metrics



