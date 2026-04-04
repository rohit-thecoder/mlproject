import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# sklearn imports for preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ================= CONFIG CLASS =================
# Yeh class preprocessing object ka path store karegi

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ================= MAIN TRANSFORMATION CLASS =================
class DataTransformation:
    def __init__(self):
        # config object initialize kar rahe hain
        self.data_transformation_config = DataTransformationConfig()

    # ---------- STEP 1: Preprocessing pipeline banana ----------
    def get_data_transformer_object(self):
        try:
            # Numerical columns (jinpe scaling lagega)
            numerical_columns = ["writing_score", "reading_score"]

            # Categorical columns (jinpe encoding lagega)
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # ---------- Numerical Pipeline ----------
            # Missing values fill + scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # missing values fill
                    ("scaler", StandardScaler())  # normalization
                ]
            )

            # ---------- Categorical Pipeline ----------
            # Missing values + encoding + scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # missing fill
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),  # convert to numbers
                    ("scaler", StandardScaler(with_mean=False))  # scaling (sparse data ke liye)
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # ---------- Combine pipelines ----------
            # ColumnTransformer alag-alag columns pe alag transformation apply karta hai
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    # ---------- STEP 2: Train & Test data transform karna ----------
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # CSV files read kar rahe hain
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            # Preprocessing object le rahe hain
            preprocessing_obj = self.get_data_transformer_object()

            # Target column (jo predict karna hai)
            target_column_name = "math_score"

            # ---------- Train data split ----------
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            # ---------- Test data split ----------
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train & test data")

            # ---------- Transformation apply ----------
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ---------- Combine features + target ----------
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            # ---------- Save preprocessor (future use ke liye) ----------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return transformed arrays + saved file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)