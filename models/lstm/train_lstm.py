import torch
import torch.optim as optim
import torch.nn as nn
from models.lstm.lstm import LSTM
from process import vocab

def initialize_model(config, device):
    model = LSTM(len(vocab), config['embedding_dim'], config['hidden_dim'], config['output_dim'], 
                  config['num_layers'], config['bidirectional'], config['dropout'])

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    return model, optimizer, criterion

def train_model(config, train_loader, wandb=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, criterion = initialize_model(config, device)
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
        if wandb: wandb.log({'loss': running_loss})
        print(f'Epoch: {i+1}, Loss: {loss.item()}')
    return model