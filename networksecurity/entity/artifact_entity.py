from dataclasses import dataclass
#It just adds acts like a decorator which probably creates a variable for an empty class.
#Let's say in my class we don't have any functions.I just need to have some variables defined, a class variables defined. We can basically use this now with respect to the data ingestion.

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class
    """
    #It is used to define the attributes of the class and their types.
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str
    invalid_train_file_path: str
    invalid_test_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

    
