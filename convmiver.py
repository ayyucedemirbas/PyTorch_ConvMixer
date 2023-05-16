import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

batch_size = 32
img_height = 128
img_width = 128
learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 25

train_path = "train/"
test_path = "test/"

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((img_height, img_width)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((img_height, img_width)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    test_path,
    transform=test_transform
)

train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

class ActivationBlock(nn.Module):
    def __init__(self, num_features):
        super(ActivationBlock, self).__init__()
        self.activation = nn.GELU()
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.activation(x)
        x = self.batch_norm(x)
        return x

class ConvStem(nn.Module):
    def __init__(self, filters, patch_size):
        super(ConvStem, self).__init__()
        self.conv = nn.Conv2d(3, filters, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvMixerBlock(nn.Module):
    def __init__(self, filters, kernel_size):
        super(ConvMixerBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=kernel_size, padding=kernel_size//2, groups=filters)
        self.pointwise_conv = nn.Conv2d(filters, filters, kernel_size=1)
        self.activation_block = ActivationBlock(filters)

    def forward(self, x):
        x0 = x
        x = self.depthwise_conv(x)
        x = self.activation_block(x + x0)
        x = self.pointwise_conv(x)
        x = self.activation_block(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, image_size, filters, depth, kernel_size, patch_size, num_classes):
        super(ConvMixer, self).__init__()
        self.image_size = image_size
        self.filters = filters
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size

        self.data_augmentation = nn.Sequential(
            transforms.ColorJitter(brightness=0.2),
)
        self.stem = ConvStem(filters, patch_size)

        self.mixer_blocks = nn.Sequential()
        for i in range(depth):
            self.mixer_blocks.add_module(f"block_{i}", ConvMixerBlock(filters, kernel_size))

        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.data_augmentation(x)
        x = x / 255.0
        x = self.stem(x)
        x = self.mixer_blocks(x)
        x = self.classification_block(x)
        return x

model = ConvMixer(image_size=img_height, filters=256, depth=8, kernel_size=5, patch_size=2, num_classes=6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "./checkpoint.pt")

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.2f}%")

    model.load_state_dict(torch.load("./checkpoint.pt"))
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            predictions += predicted.tolist()
            labels += labels.tolist()

    test_loss /= len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total

    print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

train(model, train_loader, val_loader, criterion, optimizer, num_epochs)

model.load_state_dict(torch.load("./checkpoint.pt"))
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
predictions = []
labels = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
    loss = criterion(outputs, targets)

    test_loss += loss.item()
    _, predicted = outputs.max(1)
    test_total += targets.size(0)
    test_correct += predicted.eq(targets).sum().item()

    predictions += predicted.tolist()
    labels += targets.tolist()
test_loss /= len(test_loader)
test_accuracy = 100.0 * test_correct / test_total

print(f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix

class_names = []
print("Confusion Matrix")
print(confusion_matrix(labels, predictions))
print("Classification Report")
print(classification_report(labels, predictions, target_names=class_names))
