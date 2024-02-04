from image_classifier import logger
from image_classifier.config.configuration import *
from image_classifier.components.prepare_model import PrepareModel

STAGE_NAME = ' Prepare Model Stage'

class PrepareModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        prepare_model_config= config.get_prepare_model_config()
        prepare_model=PrepareModel(prepare_model_config)
        prepare_model.get_base_model()
        prepare_model.update_base_model()

if __name__ == '__main__':
    try:
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  started <<<<<<<<<<<<<<<')
        obj= PrepareModelPipeline()
        obj.main()
        logger.info(f'>>>>>>>>>>>> {STAGE_NAME}  completed <<<<<<<<<<<<<<<')
    except Exception as e:
        logger.exception(e)