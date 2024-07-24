from models.lstm.lstm import LSTM
from models.transformer.transformer import Transformer#
from process import vocab, test_loader
import torch
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
from matplotlib import pyplot as plt
import yaml

with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if config['model'] == 'LSTM': model = LSTM(len(vocab), 100, 256, 1, 2, True, 0.5)
elif config['model'] == 'Transformer': model = Transformer(
        len(vocab),
        config['embedding_dim'],
        config['num_heads'],
        config['num_layers'],
        config['dropout']
    )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.load_state_dict(torch.load(f'{model}_model.pth'))
model.eval()

correct = 0
total = 0

y_true = []
y_pred = []
y_scores = []
if config['model'] == 'Transformer':
    with torch.no_grad():
        for text, label, length in test_loader:
            text, label = text.to(device), label.to(device)
            src_key_padding_mask = (text == 0)
            output = model(text, src_key_padding_mask=src_key_padding_mask).squeeze(1)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(output)
            
            # Convert probabilities to binary predictions
            predictions = (probs >= 0.5).long()
            
            correct += (predictions == label).sum().item()
            total += label.size(0)
            y_true.extend(label.tolist())
            y_pred.extend(predictions.tolist())
            y_scores.extend(probs.tolist())

elif config['model'] == "Transformer":
    with torch.no_grad():
        for text, label, length in test_loader:
            text, label, length = text.to(device), label.to(device), length.to(device)
            output = model(text, length).squeeze(1)
            predictions = torch.round(torch.sigmoid(output))
            correct += (predictions == label).sum().item()
            total += label.size(0)
            y_true.extend(label.tolist())
            y_pred.extend(predictions.tolist())
            y_scores.extend(torch.sigmoid(output).tolist())
            
        
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

#confusiuon matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix')

roc_auc = roc_auc_score(y_true, y_scores)
print(f'ROC AUC Score: {roc_auc:.2f}')

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_true, y_scores)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('ROC_curve')
