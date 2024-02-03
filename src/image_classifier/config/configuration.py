import os
from image_classifier.entity.config_entity import *
from image_classifier.constants import *
from image_classifier.utils.common import create_directories, read_yaml

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