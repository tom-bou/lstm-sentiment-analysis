import torch
import torch.optim as optim
import torch.nn as nn
import wandb
from model import LSTMModel
from process import vocab, train_loader, test_loader

import yaml

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

wandb.init(project='lstm-imdb', config=config, entity='tomboustedt')
model = LSTMModel(len(vocab), config['embed_dim'], config['hidden_dim'], config['output_dim'], 
                  config['n_layers'], config['bidirectional'], config['dropout'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'])

for i in range(config['num_epochs']):
    model.train()
    running_loss = 0.0
    for texts, label, lengths in train_loader:
        text, label, lengths = text.to(device), label.to(device), lengths.to(device)
        optimizer.zero_grad()
        output = model(texts, lengths).squeeze(1)
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    wandb.log({'loss': running_loss})
    print(f'Epoch: {i+1}, Loss: {loss.item()}')

correct = 0
total = 0
for label, text in test_loader:
    text, label = text.to(device), label.to(device)
    with torch.no_grad():
        output = model(text)
        loss = criterion(output, label)
        
        
    print(f'Test Loss: {loss.item()}')