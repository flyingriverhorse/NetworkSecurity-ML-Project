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