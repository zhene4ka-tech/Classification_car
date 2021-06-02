from data_loader.data_loader import DataLoader
from models.model import Model
from trainers.trainer import ModelTrainer
from utils.config import get_config_from_json

if __name__ == "__main__": #создаем точку доступа и для безопасности (зеленая кнопка появляется)
    data_loader_config = get_config_from_json("configs/data_loader_config.json")
    model_config = get_config_from_json("configs/model_config.json")
    trainer_config = get_config_from_json("configs/trainer_config.json")
    data_loader = DataLoader(data_loader_config)
    model = Model(model_config).get_model()
    trainer = ModelTrainer(trainer_config)
    trainer.train(model, data_loader)
