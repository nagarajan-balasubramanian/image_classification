import os
from image_classifier import logger
from image_classifier.entity.config_entity import ModelEvaluationConfig
import keras
from pathlib import Path
from image_classifier.utils.common import save_json
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config =config

    def evaluation(self):
        self.model = self.load_model(self.config.Model_Path)
        self.train_valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            logger.info('Train  Generator')  
            train_datagenerator = valid_datagenerator
            self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
            
    @staticmethod
    def load_model(modelpath: Path):
        return keras.models.load_model(modelpath)
    

    def evaluate_model(self):
        self.model = self.load_model(self.config.Model_Path)
        self.train_valid_generator()
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores ={"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path =Path("scores.json"), data=scores)


    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")



