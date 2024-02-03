import os
from image_classifier import logger
import gdown
import zipfile
from image_classifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config = config


    def download_data(self) -> str:
        """
        Fetch the data from the Google Drive link
        """
        try:
            download_url = self.config.source_url
            zip_download_dir = self.config.local_data_file

            logger.info(f'Downloading files from {download_url} the directory {zip_download_dir}')
            file_id =download_url.split('/')[-2]
            prefix='https://drive.google.com/uc?export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)
            
            logger.info(f'Downloaded data from {download_url} the directory {zip_download_dir} ')
        except Exception as e:
            raise e

    def unzip_file_content(self):
        """
        Unzips the file downloaded in the local drive
        """
        try:
            unzip_data_dir = self.config.unzip_data_dir
            os.makedirs(unzip_data_dir,exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file,'r') as zipref:
                zipref.extractall(unzip_data_dir)
            
            logger.info(f'Unzipping the file {self.config.local_data_file} content completed')
        except Exception as e:
            raise e