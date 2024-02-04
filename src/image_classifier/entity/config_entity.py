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
