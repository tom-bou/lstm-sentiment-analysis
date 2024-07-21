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
model = LSTMModel(len(vocab), config['embedding_dim'], config['hidden_dim'], config['output_dim'], 
                  config['num_layers'], config['bidirectional'], config['dropout'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

for i in range(config['num_epochs']):
    model.train()
    running_loss = 0.0
    for text, label, length in train_loader:
        text, label, length = text.to(device), label.to(device), length.to(device)
        optimizer.zero_grad()
        output = model(text, length).squeeze(1)
        loss = criterion(output, label)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    wandb.log({'loss': running_loss})
    print(f'Epoch: {i+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'model.pth')
wandb.save('model.pth')

correct = 0
total = 0

model.eval()  

with torch.no_grad():
    for text, label, length in test_loader:
        text, label, length = text.to(device), label.to(device), length.to(device)
        
        output = model(text, length).squeeze(1)
        
        predictions = torch.round(torch.sigmoid(output))
        
        correct += (predictions == label).sum().item()
        
        total += label.size(0)

accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')
        
print(f'Accuracy: {accuracy}')
wandb.log({'accuracy': accuracy})