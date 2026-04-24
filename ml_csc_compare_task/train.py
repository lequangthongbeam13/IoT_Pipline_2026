import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import json
import platform

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

#Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

#Training
EPOCHS = 5
epoch_log = []
total_start = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    t0 = time.time()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()

    acc = correct / len(test_data) * 100
    elapsed = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}% | Time: {elapsed:.1f}s")
    epoch_log.append({"epoch": epoch, "loss": round(total_loss/len(train_loader), 4),
                       "accuracy": round(acc, 2), "time_sec": round(elapsed, 1)})

total_time = round(time.time() - total_start, 1)

#Save Results
results = {
    "environment": "local",
    "device": str(device),
    "platform": platform.node(),
    "total_training_time_sec": total_time,
    "epochs": epoch_log,
    "final_accuracy": epoch_log[-1]["accuracy"]
}

with open("result/results_local.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDone! Total: {total_time}s | Final Accuracy: {results['final_accuracy']}%")
print("Results saved to result/results_local.json")