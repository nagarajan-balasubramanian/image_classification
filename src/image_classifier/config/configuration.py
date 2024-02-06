import os
from image_classifier.entity.config_entity import *
from image_classifier.constants import *
from image_classifier.utils.common import create_directories, read_yaml
from pathlib import Path

class ConfigurationManager:
    def __init__(self,
        config_file_path = CONFIG_FILE_PATH,
        param_file_path = PARAMS_FILE_PATH):

        self.config=read_yaml(config_file_path)
        self.params=read_yaml(param_file_path)

        create_directories([self.config['artifacts_root']])


    def get_data_ingestion_config(self) ->DataIngestionConfig:
        config = self.config['data_ingestion']

        create_directories([config['root_dir']])

        data_ingestion_config= DataIngestionConfig(config['root_dir'], 
                                                   config['source_url'],
                                                   config['local_data_file'],
                                                   config['unzip_data_dir'])
        return data_ingestion_config
    
    def get_prepare_model_config(self) ->PrepareModelConfig:
        config = self.config['prepare_model']

        create_directories([config['root_dir']])

        prepare_model_config= PrepareModelConfig(config['root_dir'], 
                                                   config['base_model_path'],
                                                   config['updated_model_path'],
                                                   self.params['IMAGE_SIZE'],
                                                   self.params['LEARNING_RATE'],
                                                   self.params['INCLUDE_TOP'],
                                                   self.params['WEIGHTS'],
                                                   self.params['CLASSES'])
        return prepare_model_config
    

    def get_model_training_config(self) ->ModelTrainingConfig:
        config = self.config['model_training']

        create_directories([config['root_dir']])

        training_data_path = os.path.join(self.config['data_ingestion']['unzip_data_dir'],
                                          "kidney-ct-scan-image")

        model_training_config= ModelTrainingConfig(config['root_dir'], 
                                                   config['trained_model_path'],
                                                   self.config['prepare_model']['updated_model_path'],
                                                   Path(training_data_path),
                                                   self.params['EPOCHS'],
                                                   self.params['BATCH_SIZE'],
                                                   self.params['AUGMENTATION'],
                                                   self.params['IMAGE_SIZE'])
        return model_training_config
    
    def get_model_evaluation_config(self) ->ModelEvaluationConfig:
        config = self.config['model_evaluation']

        model_evaluation_config= ModelEvaluationConfig(config['model_path'],
                                                   config['training_data'],
                                                   config['mlflow_uri'],
                                                   all_params=self.params,
                                                   params_batch_size=self.params['BATCH_SIZE'],
                                                   params_image_size=self.params['IMAGE_SIZE'],
                                                   params_is_augmentation=self.params['AUGMENTATION'])
        return model_evaluation_config