from image_classifier import logger
from image_classifier.config.configuration import *
from image_classifier.components.data_ingestion import DataIngestion

STAGE_NAME = ' Data Ingestion Stage'

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        data_ingestion_config= config.get_data_ingestion_config()
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.unzip_file_content()


if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  started <<<<<<<<<<<<<<<')
        obj= DataIngestionPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  completed <<<<<<<<<<<<<<<')
    except Exception as e:
        logger.exception(e)