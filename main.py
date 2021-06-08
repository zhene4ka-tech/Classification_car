from data_loader.data_loader import DataLoader
from models.model import Model
from trainers.trainer import ModelTrainer
from utils.config import get_config_from_json
from utils.get_args import get_args

if __name__ == "__main__": #создаем точку доступа и для безопасности (зеленая кнопка появляется)
    args = get_args().config
    data_loader_config = get_config_from_json("{}/data_loader_config.json".format(args))
    model_config = get_config_from_json("{}/model_config.json".format(args))
    trainer_config = get_config_from_json("{}/trainer_config.json".format(args))
    data_loader = DataLoader(data_loader_config)
    model = Model(model_config).get_model()
    trainer = ModelTrainer(trainer_config)
    trainer.train(model, data_loader)

