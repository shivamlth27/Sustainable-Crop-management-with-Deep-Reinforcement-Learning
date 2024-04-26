import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\Ayush\OneDrive\Desktop\AI-search-algo\ai-project\datasets\Crop_recommendation.csv')

df = df.dropna()

X = df.drop(['label'], axis=1)  
y = df['label']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)

class QLearningModel(nn.Module):
    def __init__(self, input_size, num_actions):
        super(QLearningModel, self).__init__()
        self.dense1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        return self.dense3(x)

model = QLearningModel(X_train.shape[1], len(label_encoder.classes_))
model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

num_epochs = 200  
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tensor)
    predicted_labels = torch.argmax(y_pred, dim=1)
    accuracy = (predicted_labels == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {accuracy:.4f}")