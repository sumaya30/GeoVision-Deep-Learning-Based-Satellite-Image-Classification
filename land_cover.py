import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch import nn, optim
from tqdm import tqdm

# ========= Config =========
DATA_DIR = os.path.join("data", "land_cover")
BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 10
LR = 1e-4  # Reduced learning rate for fine-tuning
MODEL_PATH = "models/resnet18_land_cover.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOTS_DIR = "plots"

# Create the plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# ========= Transforms =========
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========= Dataset =========
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes

# Train/Test Split (80/20)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ========= Model =========
def get_model(num_classes):
    model = models.resnet18(weights='IMAGENET1K_V1')  # Use pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ========= Training =========
def train_model(model, dataloader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss / len(dataloader), acc=100. * correct / total)

        # Store metrics for plotting
        train_losses.append(running_loss / len(dataloader))
        train_accuracies.append(100. * correct / total)
    
    torch.save(model.state_dict(), MODEL_PATH)
    print("Training complete. Model saved to:", MODEL_PATH)
    return train_losses, train_accuracies

# ========= Evaluation =========
def evaluate_model(model, dataloader, class_names):
    model.to(DEVICE)
    model.eval()
    y_true = []
    y_pred = []
    test_losses = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            
            # Loss for validation graph
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_losses.append(loss.item())
            
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate and print metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(15, 15))  # Larger figure for readability
    disp.plot(xticks_rotation=45, cmap='YlGnBu', ax=ax)
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()
    
    return test_losses, acc

# ========= Visualization =========
def visualize_predictions(model, dataloader, class_names, n=6):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images[:n].to(DEVICE), labels[:n].to(DEVICE)
    with torch.no_grad():
        outputs = model(images)
        _, preds = outputs.max(1)

    # Undo normalization for correct image display
    images = images.cpu().permute(0, 2, 3, 1) * torch.tensor([0.229, 0.224, 0.225])
    images += torch.tensor([0.485, 0.456, 0.406])
    images = np.clip(images.numpy(), 0, 1)

    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        plt.axis("off")
    plt.savefig(os.path.join(PLOTS_DIR, "sample_predictions.png"))
    plt.show()

# ========= Main =========
if __name__ == "__main__":
    model = get_model(num_classes=len(class_names))
    train_losses, train_accuracies = train_model(model, train_loader) 
    test_losses, test_accuracy = evaluate_model(model, test_loader, class_names)
    
    # Plotting Accuracy and Loss curves
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, EPOCHS + 1), [test_accuracy] * EPOCHS, 
             label='Validation Accuracy (Avg)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy vs. Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), [np.mean(test_losses)] * EPOCHS, 
             label='Validation Loss (Avg)', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_loss_graphs.png"))
    plt.show()

    visualize_predictions(model, test_loader, class_names)
