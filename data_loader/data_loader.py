import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, config):
        self.generator= ImageDataGenerator(
            rescale= config.rescale,
            rotation_range=config.rotation_range,
            brightness_range=config.brightness_range,
            horizontal_flip=config.horizontal_flip,
            fill_mode=config.fill_mode,
            zoom_range=config.zoom_range,
            validation_split=config.validation_split
        )
        directory=config.directory
        seed=config.seed
        batch_size=config.batch_size
        target_size=config.target_size
        self.train_data=self.generator.flow_from_directory(
            directory=directory,
            batch_size=batch_size,
            target_size=target_size,
            seed=seed,
            shuffle= True,
            subset="training"
        )
        self.valid_data=self.generator.flow_from_directory(
            directory=directory,
            batch_size=batch_size,
            target_size=target_size,
            seed=seed,
            shuffle=False,
            subset="validation"
        )
    def get_train_data(self):
        return self.train_data

    def get_valid_data(self):
        return self.valid_data

