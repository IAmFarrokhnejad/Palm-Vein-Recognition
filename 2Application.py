import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import os

# Author: Morteza Farrokhnejad

# Set device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the AlexNetCustom model (must match the architecture used during training)
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

# Ensure the image is in RGB format
def ensure_rgb(img):
    return img.convert('RGB') if img.mode != 'RGB' else img

# Define image preprocessing transforms (consistent with training)
input_size = 224
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.Lambda(ensure_rgb),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved model
def load_model(model_path, num_classes):
    model = AlexNetCustom(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Threshold for rejecting impostors (tune this on a validation set)
THRESHOLD = 0.7

# Predict the label for an input image, with rejection if confidence < THRESHOLD
def predict_with_reject(model, image_path, threshold=THRESHOLD):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        logits = model(image)                     # shape: [1, num_classes]
        probs = F.softmax(logits, dim=1)           # shape: [1, num_classes]
        max_prob, pred_class = torch.max(probs, 1)
        max_prob = max_prob.item()
        pred_idx = pred_class.item()

    if max_prob < threshold:
        return -1, max_prob   # -1 indicates “reject/impostor”
    else:
        return pred_idx, max_prob

# Main function with command-line arguments
def main():
    parser = argparse.ArgumentParser(
        description="Predict the label of an image using the corresponding model, with impostor rejection."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=["DB_Vein", "FYODB"],
        help="Dataset name: 'DB_Vein' or 'FYODB'"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image (e.g., image.png)"
    )
    args = parser.parse_args()

    # Define the number of classes for each dataset
    dataset_classes = {
        "DB_Vein": 98,
        "FYODB": 160
    }

    num_classes = dataset_classes[args.dataset]
    model_path = f"{args.dataset}_model.pth"

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please ensure the model is in the current directory.")
        return

    # Load the model
    model = load_model(model_path, num_classes)

    # Predict the label (or reject as impostor)
    pred_idx, conf = predict_with_reject(model, args.image)
    if pred_idx < 0:
        print(f"Impostor detected (max_prob = {conf:.3f} < {THRESHOLD})")
    else:
        print(f"Predicted Label: {pred_idx + 1} (confidence = {conf:.3f})")

if __name__ == "__main__":
    main()
