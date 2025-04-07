from networksecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging 
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from networksecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            dataframe=pd.read_csv(file_path)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data Frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                logging.info("Number of columns are valid")
                return True
            else:
                return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                #ks_2samp is a statistical test that compares two samples to determine if they come from the same distribution.
                #It returns two values: the test statistic and the p-value.
                #The p-value indicates the probability of observing the data if the null hypothesis is true.
                #If the p-value is less than the threshold, it suggests that the two samples are likely from different distributions.
                #If the p-value is greater than the threshold, it suggests that the two samples are likely from the same distribution.
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            ##It gets the directory path of the drift report file and creates the directory if it doesn't exist.
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            ##It writes the drift report to a YAML file.
            write_yaml_file(file_path=drift_report_file_path,content=report,replace=True)
            logging.info(f"Drift report file path: {drift_report_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_dataframe=DataValidation.read_data(train_file_path)
            test_dataframe=DataValidation.read_data(test_file_path)

            ##validate the number of columns
            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message=f"Train dataframe does not contain all the required columns.\n"
            #test dataframe validation
            status=self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message=f"Train dataframe does not contain all the required columns.\n"
            
            #check if there is non numeric data in the dataframe
            if not train_dataframe.select_dtypes(include=['object']).empty:
                error_message=f"{error_message} Train dataframe contains non-numeric data.\n"
            if not test_dataframe.select_dtypes(include=['object']).empty:
                error_message=f"{error_message} Train dataframe contains non-numeric data.\n"
            
            #Lets check datadrift
            status=self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            #It creates the directory for the valid train and test file paths if it doesn't exist.
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)

            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)
            ##If the data validation is successful, it creates a DataValidationArtifact object with the validation status and file paths.
            ##It also creates the directory for the valid train and test file paths if it doesn't exist.
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    

    