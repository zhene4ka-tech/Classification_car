from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from models.model_setup import transfer_models

class Model:
    def __init__(self,config):
        self.model= Sequential()
        base_model= transfer_models[config.base_model](
            include_top=config.include_top, weights='imagenet',
            input_shape=config.input_shape, pooling=None
        )

        n_layers = len(config.units)
        dense_units = config.units
        dense_activation = config.activation
        dropout_rates = config.rates
        n_classes = config.n_classes

        self.model.add(base_model)
        self.model.add( GlobalAveragePooling2D())
        for unit, rates in zip(dense_units, dropout_rates):
            self.model.add(Dense(unit, activation= dense_activation))
            self.model.add(Dropout(rates))
            self.model.add(BatchNormalization())
        self.model.add(Dense(n_classes, activation='softmax'))

    def get_model(self):
       return self.model

    def load(self, file_path):
        self.model.load_weights(file_path)