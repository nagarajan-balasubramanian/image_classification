from image_classifier import logger
from image_classifier.config.configuration import *
from image_classifier.components.model_evaluation import ModelEvaluation

STAGE_NAME = ' Model Evaluation Stage'

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        model_evaluation_config= config.get_model_evaluation_config()
        model_evaluation_config=ModelEvaluation(model_evaluation_config)
        model_evaluation_config.evaluation()
        model_evaluation_config.log_into_mlflow()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  started <<<<<<<<<<<<<<<')
        obj= ModelEvaluationPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  completed <<<<<<<<<<<<<<<')
    except Exception as e:
        logger.exception(e)