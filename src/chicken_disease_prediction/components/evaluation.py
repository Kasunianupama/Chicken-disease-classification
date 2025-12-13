from pathlib import Path
import tensorflow as tf
from chicken_disease_prediction.entity.config_entity import ValidationConfig as EvaluationConfig
from chicken_disease_prediction.utils.common import save_json



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1.0/255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=str(self.config.training_data),
            subset="validation",
            shuffle=False,
            class_mode="sparse",   # return integer labels to match sparse loss used in training
            **dataflow_kwargs,
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path, compile=False)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        # compile model with same loss/metrics used during training
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
        self._valid_generator()
        # evaluate using the loaded model instance
        self.score = self.model.evaluate(self.valid_generator, verbose=1)

    
    def save_score(self):
        scores = {"loss": float(self.score[0]), "accuracy": float(self.score[1])}
        save_json(path=Path("scores.json"), data=scores)



