from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir:Path
    source_url:str
    local_data_file:Path
    unzip_data_dir: Path

@dataclass
class PrepareModelConfig:
    root_dir:Path
    base_model:Path
    updated_base_model:Path
    params_image_size:list
    params_learning_rate:float
    params_include_top:bool
    params_weights:str
    params_classes:int

@dataclass
class ModelTrainingConfig:
    root_dir:Path
    trained_model:Path
    updated_base_model:Path
    training_data_path :Path
    params_epochs:str
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size : list

@dataclass
class ModelEvaluationConfig:
    Model_Path:Path
    training_data:Path
    mlflow_uri :str
    all_params:dict
    params_batch_size: int
    params_image_size: list
    params_is_augmentation: bool
