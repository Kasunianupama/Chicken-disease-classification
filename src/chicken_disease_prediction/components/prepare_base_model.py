import os
import urllib.request as request
from zipfile import ZipFile 
import tensorflow as tf
from chicken_disease_prediction import logger
from chicken_disease_prediction.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None
        self.full_model = None

    def get_base_model(self):
        try:
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=self.config.params_image_size,
                include_top=self.config.params_include_top,
                weights=self.config.params_weights,
                classes=self.config.params_classes,
            )
            base_model_path = Path(self.config.base_model_path)
            base_model_path.parent.mkdir(parents=True, exist_ok=True)
            base_model.save(str(base_model_path))
            self.model = base_model
            print(f"Base model saved at : {base_model_path}")
        except Exception as e:
            raise e

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return full_model

    def update_base_model(self):
        if self.model is None:
            raise RuntimeError("Base model not loaded. Call get_base_model() before update_base_model().")
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=False,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate,
        )
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(p))
        print(f"Updated base model is saved at : {p}")