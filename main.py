from image_classifier import logger
from image_classifier.pipeline.stage01_data_ingestion import DataIngestionPipeline
from image_classifier.pipeline.stage02_prepare_model import PrepareModelPipeline
from image_classifier.pipeline.stage03_model_training import ModelTariningPipeline
from image_classifier.pipeline.stage04_model_evaluation import ModelEvaluationPipeline
import os

os.environ['MLFLOW_TRACKING_USERNAME']='nagarajan.bala'
os.environ['MLFLOW_TRACKING_PASSWORD']='23cbda8f047164298e08463268d2d17f3edd7f1b'

logger.info('Logging initiated')

# STAGE_NAME = 'Data Ingestion Stage'        
        
# try:
#     logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
#     data_ingestion =DataIngestionPipeline()
#     data_ingestion.main()
# except Exception as e:
#     logger.exception(e)


# STAGE_NAME = 'Prepare Model Stage'        
        
# try:
#     logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
#     prepare_model =PrepareModelPipeline()
#     prepare_model.main()
# except Exception as e:
#     logger.exception(e)


# STAGE_NAME = 'Model Training Stage'        
        
# try:
#     logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
#     model_training =ModelTariningPipeline()
#     model_training.main()
# except Exception as e:
#     logger.exception(e)

STAGE_NAME = 'Model Evaluation Stage'        
        
try:
    logger.info(f'>>>>>>>>>>>> {STAGE_NAME}   <<<<<<<<<<<<<<<')
    model_evaluation =ModelEvaluationPipeline()
    model_evaluation.main()
except Exception as e:
    logger.exception(e)

