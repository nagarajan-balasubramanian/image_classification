import os
from image_classifier import logger
from image_classifier.entity.config_entity import ModelTrainingConfig
import urllib.request as request
import keras
from pathlib import Path

class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config =config

    
    def get_base_model(self):
        self.model = keras.models.load_model(self.config.updated_base_model)

    
    @staticmethod
    def save_model(modelpath:Path, model:keras.Model):
        model.save(modelpath)

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
            directory=self.config.training_data_path,
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
            directory=self.config.training_data_path,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def train(self):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            modelpath=self.config.trained_model,
            model=self.model)