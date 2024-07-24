import torch
import wandb
from process import train_loader, test_loader
from models.lstm.train_lstm import train_model as train_lstm
from models.transformer.train_transformer import train_model as train_transformer
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

wandb.init(project='imdb_review', config=config, entity='tomboustedt')

if config['model'] == 'LSTM':
    model = train_lstm(config, train_loader, wandb)
elif config['model'] == 'Transformer':
    model = train_transformer(config, train_loader, wandb)

torch.save(model.state_dict(), f'{config["model"]}_model.pth')
wandb.save(f'{config["model"]}_model.pth')
