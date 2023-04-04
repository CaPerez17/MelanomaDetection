import torch
import torchvision
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a melanoma detection model')
parser.add_argument('--data_dir', type=str, default='data',
                    help='path to directory containing data')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for optimizer')
parser.add_argument('--model_path', type=str, default='models/model.pt',
                    help='path to save trained model')
args = parser.parse_args()

# Load data
train_dir = os.path.join(args.data_dir, 'train')
val_dir = os.path.join(args.data_dir, 'val')
test_dir = os.path.join(args.data_dir, 'test')
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=torchvision.transforms.ToTensor())
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=torchvision.transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Define model
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Train model
for epoch in range(args.num_epochs):
    train_loss = 0.0
    train_correct = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
    train_acc = train_correct / len(train_dataset)
    train_loss = train_loss / len(train_loader)
    
    val_loss = 0.0
    val_correct = 0
    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / len(val_dataset)
    val_loss = val_loss / len(val_loader)
    
    print(f'Epoch {epoch+1} train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
    
# Save model
os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
torch.save(model.state_dict(), args.model_path)
