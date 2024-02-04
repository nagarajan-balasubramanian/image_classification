import os
from image_classifier import logger
from image_classifier.entity.config_entity import PrepareModelConfig
import urllib.request as request
import keras
from pathlib import Path

class PrepareModel:
    def __init__(self,config:PrepareModelConfig):
        self.config =config

    
    def get_base_model(self):
        self.model = keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(modelpath=self.config.base_model, model=self.model)

    @staticmethod
    def save_model(modelpath:Path, model:keras.Model):
        model.save(modelpath)

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = keras.layers.Flatten()(model.output)
        prediction = keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(modelpath=self.config.updated_base_model, model=self.full_model)