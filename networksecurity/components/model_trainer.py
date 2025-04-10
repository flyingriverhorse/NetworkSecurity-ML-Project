import os
import sys

from networksecurity.exception.exception import NetworkSecurityException 
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from urllib.parse import urlparse

import dagshub
dagshub.init(repo_owner='flyingriverhorse', repo_name='NetworkSecurity-ML-Project', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/krishnaik06/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="krishnaik06"
os.environ["MLFLOW_TRACKING_PASSWORD"]="7104284f1bb44ece21e0e2adb4e36a250ae3251f"


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,
                 data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e
        
    def track_mlflow(self,best_model,classificationmetric):
        mlflow.set_registry_uri("https://dagshub.com/flyingriverhorse/NetworkSecurity-ML-Project.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run():
            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_metric("f1_score", classificationmetric.f1_score)
            mlflow.log_metric("recall_score", classificationmetric.recall_score)
            mlflow.log_metric("precision_score", classificationmetric.precision_score)
            mlflow.log_param("best_model_name", best_model.__class__.__name__)
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(best_model, "model", registered_model_name=best_model)
            else:
                mlflow.sklearn.log_model(best_model, "model")

            # Log the mlflow tracking URI for the model
            url = urlparse(mlflow.get_tracking_uri())
            print(f"Model tracking URI: {url.scheme}://{url.netloc}{url.path}")
        
    def train_model(self,X_train,y_train,x_test,y_test)->NetworkModel:
        models = {
                "KNeighbors Classifier": KNeighborsClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            "KNeighbors Classifier": {
                'n_neighbors':[3,5,7,9,11],
                'weights':['uniform','distance'],
                # 'algorithm':['auto','ball_tree','kd_tree','brute'],
                # 'leaf_size':[10,20,30,40],
            },
            "Logistic Regression": {
                'penalty':['l1','l2'],
                'C':[0.001,0.01,0.1,1,10],
                # 'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                # 'max_iter':[100,200,300],
            },
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            }
            
        }
        model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                          models=models,param=params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))
        logging.info(f"Best model score: {best_model_score}")   
        ## To get best model name from dict
        ## and then get the model object from models dict
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]
        y_train_pred=best_model.predict(X_train)
        logging.info(f"Best model name: {best_model_name}")
        logging.info(f"Best model: {best_model}")
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)
        
        ## Track the experiements with mlflow
        self.track_mlflow(best_model,classification_train_metric)
        
        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        self.track_mlflow(best_model,classification_test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)

        #we use pickel file to save the model
        Network_Model=NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path,obj=NetworkModel)
        #model pusher
        save_object("final_model/model.pkl",best_model)
        

        ## Model Trainer Artifact
        model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")
        return model_trainer_artifact

        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            #load numpy array data
            train_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_array = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1], # not last column
                train_array[:,-1], # taking last column
                test_array[:,:-1], # not last column
                test_array[:,-1] # taking last column
            )

            model=self.train_model(x_train,y_train,x_test,y_test)

        
        except Exception as e:
            raise NetworkSecurityException(e,sys) from e