import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import wandb

# Define the Classifier model
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data
wbcd = load_breast_cancer()
feature_names = wbcd.feature_names
labels = wbcd.target_names

X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 2  # binary classification

# Initialize the Classifier
model = Classifier(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    _, predicted = torch.max(y_pred_tensor, 1)
    correct = (predicted == y_test_tensor).sum().item()
    total = y_test_tensor.size(0)
    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy:.2%}')

# ROC Curve
y_probas = model(X_test_tensor).softmax(dim=1).detach().numpy()
fpr, tpr, _ = roc_curve(y_test, y_probas[:, 1])
roc_auc = auc(fpr, tpr)

# WandB initialization
wandb.init(project='my-scikit-integration', name="classification")

# WandB sklearn plots
wandb.sklearn.plot_class_proportions(y_train, y_test, labels)

# Convert PyTorch tensors to NumPy arrays for sklearn plots
X_train_numpy, X_test_numpy, y_train_numpy, y_test_numpy = (
    X_train.numpy(), X_test.numpy(), y_train.numpy(), y_test.numpy()
)

wandb.sklearn.plot_learning_curve(model, X_train_numpy, y_train_numpy)

wandb.sklearn.plot_roc(y_test_numpy, y_probas, labels)

wandb.sklearn.plot_precision_recall(y_test_numpy, y_probas, labels)

# Feature importances plot for RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_numpy, y_train_numpy)
wandb.sklearn.plot_feature_importances(model_rf, feature_names)

# Classifier plot for RandomForestClassifier
wandb.sklearn.plot_classifier(model_rf, 
                              X_train_numpy, X_test_numpy, 
                              y_train_numpy, y_test_numpy, 
                              model_rf.predict(X_test_numpy), model_rf.predict_proba(X_test_numpy), 
                              labels, 
                              is_binary=True, 
                              model_name='RandomForest')

# Finish WandB run
wandb.finish()
