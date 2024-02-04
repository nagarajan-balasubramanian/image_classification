from image_classifier import logger
from image_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from image_classifier.pipeline.stage02_prepare_model import PrepareModelPipeline

logger.info('Logging initiated')

STAGE_NAME = 'Data Ingestion Stage'        
        
try:
    logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
    data_ingestion =DataIngestionPipeline()
    data_ingestion.main()
except Exception as e:
    logger.exception(e)


STAGE_NAME = 'Prepare Model Stage'        
        
try:
    logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
    prepare_model =PrepareModelPipeline()
    prepare_model.main()
except Exception as e:
    logger.exception(e)





