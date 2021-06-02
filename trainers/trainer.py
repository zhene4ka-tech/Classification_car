from trainers.trainer_setup import optimisers, plot_history
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class ModelTrainer:
    def __init__(self, config):
        self.learning_rate= config.learning_rate
        self.epochs= config.epochs
        self.loss_function= config.loss_function
        self.metrics= config.metrics
        self.optimizer= optimisers[config.optimiser](self.learning_rate)
        self.batch_size= config.batch_size
        self.callbacks= [
            ModelCheckpoint(config.checkpoint_path, monitor= config.checkpoint_monitor, verbose=1),
            EarlyStopping(patience=config.early_patience, monitor=config.early_monitor),
            ReduceLROnPlateau(patience=config.reduce_patience, monitor=config.reduce_monitor)
        ]
    def train(self, model, data_loader):
        model.compile(optimizer= self.optimizer, loss= self.loss_function, metrics=self.metrics)
        data_train= data_loader.get_train_data()
        data_val= data_loader.get_valid_data()
        for i in range(self.epochs):
            history= model.fit(
                data_train,
                epochs=1,
                batch_size= self.batch_size,
                callbacks= self.callbacks,
                validation_data= data_val
                                    )
            plot_history(history)


