from src.full_model.training.model_training import ModelTraining
from src.full_model.loader.data_loder import DataLoader
from src.full_model.features.feature_eng import FeatureEngineering
import sklearn
from sklearn.model_selection import train_test_split
import logging


# setup logging

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def main():

    try:

        # 1.load_data

        logger.info("loding data")
        loader = DataLoader(file_path='data/clean_data/titanic_clean_data.csv')
        df = loader.Load_data()
        loader.validate_data(df,required_columns=['survived','pclass',
                                                  'sex','age','sibsp','parch','fare'])
        
        # 2 feature engineering
        logger.info("feature engineering")

        fe = FeatureEngineering()
        df = fe.handle_missing_values(df)
        df = fe.encode_categorical(df)


        # 3 split data
        logger.info("splitting data into train and test")
        X = df.drop('survived',axis=1)
        y = df['survived']

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

        #4 scale features
        logger.info("scaling features")
        X_train_scaled,X_test_scaled = fe.scale_features(X_train,X_test)

        #5  train model
        logger.info("training model")
        trainer = ModelTraining(model_type="RandomForest",
                                n_estimators=100,max_depth=5,random_state=42)
        
        trainer = ModelTraining(model_type="LogisticRegression", max_iter=1000, random_state=42)

        trainer = ModelTraining(model_type="DecisionTree", max_depth=5, random_state=42)

        trainer.train_model(X_train_scaled, y_train)
        
        # 6 Evaluate
        logger.info("Evaluating model")
        metrics = trainer.evaluate(X_test_scaled,y_test)
        for metric,score in metrics.items():
            logger.info(f"{metric}: {score}")

        logger.info("Comparing models")
        trainer.compare_models(X_train_scaled, y_train, X_test_scaled, y_test)

        
        # save model
        logger.info("saving model")
        trainer.save_model(model_path=f'models/{trainer.model_type}.pkl')
        logger.info("model saved successfully")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise e
    
if __name__ == "__main__":
    main()