import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:

    def __init__(self,file_path:str):
        self.file_path = Path(file_path)

    def Load_data(self)->pd.DataFrame:

        if not self.file_path.exists():
            raise FileNotFoundError(f"The file {self.file_path} does not exist.")
        
        df = pd.read_csv(self.file_path)
        return df
    

    def validate_data(self,df:pd.DataFrame, required_columns:list[str] | None = None)->bool:
        if df.empty:
            raise ValueError("The DataFrame is empty.")
        
        if df.isnull().sum().sum() > 0:
            raise ValueError("The DataFrame contains missing values.")
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"The DataFrame is missing required columns: {missing_columns}")
        
        return True