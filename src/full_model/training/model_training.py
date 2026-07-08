import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import joblib
from typing import Dict,Any

class ModelTraining:

    def __init__(self,model_type:str='RandomForest',**model_params):
        self.model_type = model_type
        self.model_params = model_params
        self.model = None
        self.training_score = {}

    
    def train_model(self,X_train,y_train)-> None:

        print(f"model type recived: {self.model_type}")

        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(**self.model_params)
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(**self.model_params)
        elif self.model_type == 'DecisionTree':
            self.model = DecisionTreeClassifier(**self.model_params)
        else:
            raise ValueError(f"Model type {self.model_type} is not supported.")

        self.model.fit(X_train,y_train)

        # cross validation

        cv_score = cross_val_score(self.model,X_train,y_train,cv=5,scoring='accuracy')
        self.training_score["cross_val_score"] = cv_score
        print(f"Cross validation score: {cv_score}")


    def evaluate(self,X_test:pd.DataFrame,y_test:pd.Series)-> Dict[str,float]:

        if self.model is None:
            raise ValueError("Model is not trained yet. Please train the model before evaluation.")
        
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred)

        self.training_score["accuracy"] = accuracy
        self.training_score["f1_score"] = f1
        self.training_score["roc_auc"] = roc_auc

        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1}")
        print(f"ROC AUC Score: {roc_auc}")

        return self.training_score
    def compare_models(self, X_train, y_train, X_test, y_test):

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        results = {}

        for name, model in models.items():

        # Train model
            model.fit(X_train, y_train)

        # Prediction
            y_pred = model.predict(X_test)

        # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

        # Cross Validation
            cv_score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy"
            ).mean()

            results[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "CV Score": cv_score
            }

        return results

        for name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            cv_score = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="accuracy"
            ).mean()

            results[name] = {
                "Accuracy": accuracy,
                "F1 Score": f1,
                "ROC AUC": roc_auc,
                "CV Score": cv_score
            }

        return results
    
    def save_model(self,model_path:str)-> None:
        joblib.dump(self.model,model_path)
        print(f"Model saved at {model_path}")

    
    @staticmethod
    def load_model(model_path:str)-> Any:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model