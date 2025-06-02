import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed hyperparameters per dataset
db_vein_params = {
    'learning_rate': 0.001,
    'batch_size': 16,
    'dropout_rate': 0.2,
    'optimizer': 'SGD'
}

fyodb_params = {
    'learning_rate': 0.005,
    'batch_size': 16,
    'dropout_rate': 0.2,
    'optimizer': 'SGD'
}

# Number of classes
num_classes_db_vein = 98
num_classes_fyodb = 160
num_epochs = 30
input_size = 224

# Helper to convert to RGB
def ensure_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(ensure_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Lambda(ensure_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Custom Dataset for DB_Vein
class DBVeinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        print(f"[DBVeinDataset] Checking root directory: {root_dir}")
        if not os.path.exists(root_dir):
            print(f"[DBVeinDataset] ERROR: '{root_dir}' does not exist.")
            return

        for subject in range(1, 99):
            subject_dir = os.path.join(root_dir, f"{subject:03d}")
            if not os.path.isdir(subject_dir):
                print(f"[DBVeinDataset] WARNING: no folder {subject_dir}")
                continue

            print(f"[DBVeinDataset] Scanning folder: {subject_dir}")
            for img_name in os.listdir(subject_dir):
                print(f"[DBVeinDataset] found file: {img_name}")
                if img_name.lower().endswith('.bmp') and ('vr' in img_name.lower() or 'vl' in img_name.lower()):
                    img_path = os.path.join(subject_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(subject - 1)
                    print(f"[DBVeinDataset] → added: {img_path}")

        print(f"[DBVeinDataset] Total images found: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Custom Dataset for FYODB
class FYODBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        print(f"[FYODBDataset] Checking root directory: {root_dir}")
        if not os.path.exists(root_dir):
            print(f"[FYODBDataset] ERROR: '{root_dir}' does not exist.")
            return

        for session in ['Session1', 'Session2']:
            session_dir = os.path.join(root_dir, session)
            if not os.path.isdir(session_dir):
                print(f"[FYODBDataset] WARNING: no folder {session_dir}")
                continue

            print(f"[FYODBDataset] Scanning folder: {session_dir}")
            for img_name in os.listdir(session_dir):
                print(f"[FYODBDataset] found file: {img_name}")
                if img_name.endswith('_L.png') or img_name.endswith('_R.png'):
                    try:
                        subject_id = int(img_name.split('_')[0]) - 1
                        img_path = os.path.join(session_dir, img_name)
                        self.images.append(img_path)
                        self.labels.append(subject_id)
                        print(f"[FYODBDataset] → added: {img_path}")
                    except ValueError:
                        print(f"[FYODBDataset] ✗ could not parse ID from: {img_name}")

        print(f"[FYODBDataset] Total images found: {len(self.images)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# Helper to wrap subset + transform
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Create DataLoaders
def create_dataloaders(dataset, batch_size, val_split=0.2, train_all=False):
    if train_all:
        # Apply train transforms to the entire dataset
        train_ds = DatasetFromSubset(Subset(dataset, list(range(len(dataset)))), data_transforms['train'])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        return {'train': train_loader}, {'train': len(dataset)}
    else:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(val_split * dataset_size))
        train_idx, val_idx = indices[split:], indices[:split]

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_ds = DatasetFromSubset(train_subset, data_transforms['train'])
        val_ds = DatasetFromSubset(val_subset, data_transforms['val'])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        return {'train': train_loader, 'val': val_loader}, {'train': len(train_subset), 'val': len(val_subset)}

# AlexNet-inspired architecture
class AlexNetCustom(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(AlexNetCustom, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(256)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Training loop with history tracking
def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    history = {
        'train_loss': [],
        'train_acc': [],
    }
    if 'val' in dataloaders:
        history['val_loss'] = []
        history['val_acc'] = []
        best_acc = 0.0
        best_wts = model.state_dict()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects.double() / dataset_sizes['train']
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        if 'val' in dataloaders:
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_corrects = 0

            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

            val_epoch_loss = val_loss / dataset_sizes['val']
            val_epoch_acc = val_corrects.double() / dataset_sizes['val']
            print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc.item())

            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_wts = model.state_dict()

        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")

    if 'val' in dataloaders:
        print(f"Best val Acc: {best_acc:.4f}")
        model.load_state_dict(best_wts)
    else:
        best_acc = epoch_acc  # final training accuracy

    return model, best_acc, history

# Function to run training for one dataset with fixed hyperparameters
def run_experiment(ds_name, ds_path, n_classes, ds_class, params):
    print(f"\n[Main] Loading dataset: {ds_name}")
    if not os.path.exists(ds_path):
        print(f"[Main] ERROR: '{ds_path}' not found")
        return

    # Initialize dataset without transforms
    dataset = ds_class(root_dir=ds_path, transform=None)
    print(f"[Main] {ds_name} size: {len(dataset)}")
    if len(dataset) == 0:
        print(f"[Main] ERROR: {ds_name} is empty")
        return

    batch_size = params['batch_size']
    lr = params['learning_rate']

    # Create dataloaders with train_all=True
    dataloaders, ds_sizes = create_dataloaders(dataset, batch_size, train_all=True)

    # Initialize model
    model = AlexNetCustom(num_classes=n_classes, dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    # Train model
    model, best_acc, history = train_model(model, criterion, optimizer, dataloaders, ds_sizes, num_epochs)

    # Save the model
    model_path = f"{ds_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Log results
    if 'val' in dataloaders:
        log_message = f"Best Validation Accuracy: {best_acc.item():.6f}"
    else:
        log_message = f"Final Training Accuracy: {best_acc.item():.6f}"

    with open(f"hyperparameter_results_{ds_name}.txt", 'a') as f:
        f.write(f"\nRun at {datetime.now()}\n")
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Model: AlexNetCustom\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Learning Rate: {lr}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Dropout Rate: {params['dropout_rate']}\n")
        f.write(f"  Optimizer: {params['optimizer']}\n")
        f.write(log_message + "\n")
        f.write("-" * 50 + "\n")

    print(f"[Main] Done: {log_message}")

    # Plot training metrics
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'{ds_name} Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'{ds_name} Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{ds_name}_training_plots.png")
    plt.show()

    # Separate accuracy vs. epoch plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy vs. Epoch for {ds_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"{ds_name}_accuracy_plot.png")
    plt.show()

def main():
    datasets = [
        ('FYODB', r'PATH TO DATA', # Specify this path.
         num_classes_fyodb, FYODBDataset, fyodb_params),
        ('DB_Vein', r'PATH TO DATA', # Specify this path.
         num_classes_db_vein, DBVeinDataset, db_vein_params),
    ]

    for ds_name, ds_path, n_classes, ds_class, params in datasets:
        run_experiment(ds_name, ds_path, n_classes, ds_class, params)

if __name__ == "__main__":
    main()


# Author: Morteza Farrokhnejad