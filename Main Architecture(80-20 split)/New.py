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
from sklearn.model_selection import train_test_split

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
num_epochs = 120 # This variable is overwritten in run_experiment based on the dataset
input_size = 224

# Helper to convert to RGB
def ensure_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img

# Enhanced data transforms with more augmentation for FYODB
data_transforms_standard = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(ensure_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Lambda(ensure_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Enhanced transforms for FYODB with more aggressive augmentation
data_transforms_fyodb = {
    'train': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.Lambda(ensure_rgb),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
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
                if img_name.endswith('_L.png') or img_name.endswith('_R.png') or img_name.endswith('.jpg'):
                    try:
                        if img_name.endswith('.jpg'):
                            subject_id = int(img_name.split('_')[0].replace("s", "")) - 1
                        else:
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

# Create DataLoaders with train-test split
def create_dataloaders(dataset, batch_size, test_split=0.2, dataset_name=None):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # Get unique labels for analysis
    labels = [dataset[i][1] for i in indices]
    unique_labels = set(labels)
    num_classes = len(unique_labels)
    
    print(f"Dataset has {dataset_size} samples across {num_classes} classes")
    
    # Check if we have enough samples for stratified split
    min_samples_per_class = min([labels.count(label) for label in unique_labels])
    required_test_samples = max(1, int(min_samples_per_class * test_split))
    
    print(f"Minimum samples per class: {min_samples_per_class}")
    print(f"Required test samples per class: {required_test_samples}")
    
    # For datasets with very few samples per class (like FYODB), use a different strategy
    if min_samples_per_class <= 2 or (dataset_size * test_split) < num_classes:
        print(f"Warning: Dataset has limited samples per class. Using per-class split strategy.")
        
        # Group indices by class
        class_indices = {}
        for idx, label in enumerate(labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        train_idx = []
        test_idx = []
        
        # For each class, split individually
        for label, class_idx in class_indices.items():
            np.random.shuffle(class_idx)
            n_class_samples = len(class_idx)
            
            if n_class_samples == 1:
                # If only one sample, put it in training
                train_idx.extend(class_idx)
            elif n_class_samples == 2:
                # If two samples, one for train, one for test
                train_idx.append(class_idx[0])
                test_idx.append(class_idx[1])
            else:
                # If more samples, use proper split
                n_test = max(1, int(n_class_samples * test_split))
                test_idx.extend(class_idx[:n_test])
                train_idx.extend(class_idx[n_test:])
        
        # Shuffle the final indices
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        
    else:
        # Use standard stratified split for datasets with sufficient samples
        try:
            train_idx, test_idx = train_test_split(
                indices, 
                test_size=test_split, 
                stratify=labels, 
                random_state=42
            )
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            # Fallback to simple random split
            np.random.shuffle(indices)
            split_point = int(dataset_size * (1 - test_split))
            train_idx = indices[:split_point]
            test_idx = indices[split_point:]
    
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    # Choose appropriate transforms based on dataset
    if dataset_name == 'FYODB':
        transforms_to_use = data_transforms_fyodb
    else:
        transforms_to_use = data_transforms_standard
    
    train_ds = DatasetFromSubset(train_subset, transforms_to_use['train'])
    test_ds = DatasetFromSubset(test_subset, transforms_to_use['test'])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    
    # Verify class distribution in test set
    test_labels = [labels[i] for i in test_idx]
    test_classes = len(set(test_labels))
    print(f"Test set contains {test_classes} unique classes out of {num_classes} total classes")
    
    return {'train': train_loader, 'test': test_loader}, {'train': len(train_subset), 'test': len(test_subset)}

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

        # Keep track of best training accuracy for model saving
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_wts = model.state_dict()

        print()

    elapsed = time.time() - since
    print(f"Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s")
    print(f"Best training Acc: {best_acc:.4f}")
    
    # Load best weights
    model.load_state_dict(best_wts)
    
    return model, best_acc, history, elapsed

# Test function
def test_model(model, test_loader, criterion, dataset_size):
    test_start_time = time.time()
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
    
    test_loss = test_loss / dataset_size
    test_acc = test_corrects.double() / dataset_size
    test_elapsed = time.time() - test_start_time
    
    print(f"Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
    print(f"Testing time: {test_elapsed:.2f}s")
    
    return test_acc, test_loss, test_elapsed

# Function to run training for one dataset with fixed hyperparameters
def run_experiment(ds_name, ds_path, n_classes, ds_class, params, num_epochs_for_dataset):
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

    # Create dataloaders with 80-20 train-test split
    dataloaders, ds_sizes = create_dataloaders(dataset, batch_size, test_split=0.2, dataset_name=ds_name)

    # Initialize model
    model = AlexNetCustom(num_classes=n_classes, dropout_rate=params['dropout_rate']).to(device)
    criterion = nn.CrossEntropyLoss()

    # Select optimizer
    if params['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {params['optimizer']}")

    # Train model
    model, best_train_acc, history, training_time = train_model(model, criterion, optimizer, dataloaders, ds_sizes, num_epochs_for_dataset)

    # Test the model
    test_acc, test_loss, test_time = test_model(model, dataloaders['test'], criterion, ds_sizes['test'])

    # Calculate total computational time
    total_time = training_time + test_time

    # Save the model
    model_path = f"{ds_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Log results
    log_message = f"Best Training Accuracy: {best_train_acc.item():.6f}, Test Accuracy: {test_acc.item():.6f}"

    with open(f"hyperparameter_results_{ds_name}.txt", 'a') as f:
        f.write(f"\nRun at {datetime.now()}\n")
        f.write(f"Dataset: {ds_name}\n")
        f.write(f"Model: AlexNetCustom\n")
        f.write("Hyperparameters:\n")
        f.write(f"  Learning Rate: {lr}\n")
        f.write(f"  Batch Size: {batch_size}\n")
        f.write(f"  Dropout Rate: {params['dropout_rate']}\n")
        f.write(f"  Optimizer: {params['optimizer']}\n")
        f.write(f"  Number of Epochs: {num_epochs_for_dataset}\n")
        f.write(f"Dataset Information:\n")
        f.write(f"  Total Classes: {n_classes}\n")
        f.write(f"  Training samples: {ds_sizes['train']}, Test samples: {ds_sizes['test']}\n")
        f.write(f"  Train/Test Split: {(1-ds_sizes['test']/(ds_sizes['train']+ds_sizes['test']))*100:.1f}%/{(ds_sizes['test']/(ds_sizes['train']+ds_sizes['test']))*100:.1f}%\n")
        f.write(log_message + "\n")
        f.write("Computational Cost (Time):\n")
        f.write(f"  Training Time: {training_time//60:.0f}m {training_time%60:.2f}s\n")
        f.write(f"  Testing Time: {test_time:.2f}s\n")
        f.write(f"  Total Time: {total_time//60:.0f}m {total_time%60:.2f}s\n")
        f.write(f"  Time per Epoch: {training_time/num_epochs_for_dataset:.2f}s\n")
        f.write(f"  Time per Training Sample: {training_time/ds_sizes['train']:.4f}s\n")
        f.write("-" * 50 + "\n")

    print(f"[Main] Done: {log_message}")
    print(f"[Main] Training Time: {training_time//60:.0f}m {training_time%60:.2f}s")
    print(f"[Main] Testing Time: {test_time:.2f}s")
    print(f"[Main] Total Time: {total_time//60:.0f}m {total_time%60:.2f}s")

    # Plot training metrics
    epochs = range(1, num_epochs_for_dataset + 1)
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title(f'{ds_name} Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.axhline(y=test_acc.item(), color='r', linestyle='--', label=f'Test Acc: {test_acc.item():.4f}')
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
    plt.axhline(y=test_acc.item(), color='r', linestyle='--', label=f'Test Acc: {test_acc.item():.4f}')
    plt.title(f'Accuracy vs. Epoch for {ds_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{ds_name}_accuracy_plot.png")
    plt.show()

def main():
    datasets = [
        ('FYODB', r'PATH TO DATA', # Specify the path
         num_classes_fyodb, FYODBDataset, fyodb_params, 30),
        ('DB_Vein', r'PATH TO DATA', # Specify the path
         num_classes_db_vein, DBVeinDataset, db_vein_params, 39),
    ]

    for ds_name, ds_path, n_classes, ds_class, params, num_epochs_for_dataset in datasets:
        run_experiment(ds_name, ds_path, n_classes, ds_class, params, num_epochs_for_dataset)

if __name__ == "__main__":
    main()


# Author: Morteza Farrokhnejad