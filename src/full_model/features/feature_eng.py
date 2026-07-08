import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder


class FeatureEngineering:

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_name = None

    
    def handle_missing_values(self,df:pd.DataFrame)->pd.DataFrame:
    
        # numerical missing values
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)

        # categorical missing values
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical(self,df:pd.DataFrame)->pd.DataFrame:
        df = pd.get_dummies(df, drop_first=True)
        return df
    

    def scale_features(self,X_train:pd.DataFrame,X_test:pd.DataFrame | None = None)->tuple:


        self.feature_name = X_train.columns.tolist()

        # fit the scaler on the training data

        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), 
                                      columns=self.feature_name)
        
        # transform the test data
        if X_test is not None:
            X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), 
                                         columns=self.feature_name)
        else:
            X_test_scaled = None
        return X_train_scaled, X_test_scaled
    
    

        
