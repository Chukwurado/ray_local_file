
class Config:
    def __init__(self, mlflow_filepath: str, ray_filepath: str, experiment_name: str, max_epochs=1):
        self.mlflow_filepath = mlflow_filepath
        self.ray_filepath = ray_filepath
        self.experiment_name = experiment_name 
        self.max_epochs = max_epochs
