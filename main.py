from image_classifier import logger
from image_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeline

logger.info('Logging initiated')

STAGE_NAME = 'Data Ingestion Stage'        
        
try:
    logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
    data_ingestion =DataIngestionPipeline()
    data_ingestion.main()
except Exception as e:
    logger.exception(e)




