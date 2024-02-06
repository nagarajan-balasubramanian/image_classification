from image_classifier import logger
from image_classifier.config.configuration import *
from image_classifier.components.model_training import ModelTraining

STAGE_NAME = ' Model Training Stage'

class ModelTariningPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        model_training_config= config.get_model_training_config()
        model_training=ModelTraining(model_training_config)
        model_training.get_base_model()
        model_training.train_valid_generator()
        model_training.train()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  started <<<<<<<<<<<<<<<')
        obj= ModelTariningPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  completed <<<<<<<<<<<<<<<')
    except Exception as e:
        logger.exception(e)