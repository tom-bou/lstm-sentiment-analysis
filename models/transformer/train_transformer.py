import torch
from models.transformer.transformer import Transformer
from process import vocab

def initialize_model(config, device):
    model = Transformer(
        len(vocab),
        config['embedding_dim'],
        config['num_heads'],
        config['num_layers'],
        config['dropout']
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
    criterion = torch.nn.BCEWithLogitsLoss()
    return model, optimizer, criterion

def train_model(config, train_loader, wandb=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, optimizer, criterion = initialize_model(config, device)
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        for texts, labels, lengths in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            src_key_padding_mask = (texts == 0)
            optimizer.zero_grad()
            outputs = model(texts, src_key_padding_mask=src_key_padding_mask).squeeze(1)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({'loss': total_loss/len(train_loader)})
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {total_loss/len(train_loader):.4f}')
    
    return model