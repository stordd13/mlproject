import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestionConfig, DataIngestion
from src.components.model_trainer import ModelTrainer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact',"preprocessor.pkl" )

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_features = ["reading score", "writing score"]
            categorical_features = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical columns standard scaling completed")


            categories = [
                            ["female", "male"],  # categories for "gender"
                            ["group A", "group B", "group C", "group D", "group E"],  # categories for "race/ethnicity"
                            ["some high school", "high school", "some college", 
                             "associate's degree", "bachelor's degree", "master's degree"],  # categories for "parental level of education"
                            ["free/reduced", "standard"],  # categories for "lunch"
                            ["none", "completed"],  # categories for "test preparation course"
]
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder(categories=categories, handle_unknown='error')
                     )
                ]
            )

            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                   ("num_pipeline", num_pipeline, numerical_features),
                   ("cat_pipeline", cat_pipeline, categorical_features)
                ]
                )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("'Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
    
            target_column_name = "math score"
            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            print(preprocessor_obj.transformers_)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_df]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj       
                        )

            return (train_arr, 
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e,sys) from e
        
if __name__ == "__main__":
    obj=DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))