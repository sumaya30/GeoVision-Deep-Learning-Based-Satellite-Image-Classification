import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support

# Paths
train_dir = r"C:\Users\DELL\Desktop\satellite_img_classification\data\disaster\train"
val_dir = r"C:\Users\DELL\Desktop\satellite_img_classification\data\disaster\val"

save_path = "models/vit_disaster.pth"
PLOTS_DIR = "plots/disaster"

# Create the plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Model setup
def create_model(num_classes):
    model = models.vit_b_16(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False  # freeze feature extractor
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

# Accuracy function
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return torch.sum(preds == labels).item() / len(labels)

# Training function
def train_model(model, train_loader, val_loader, epochs=15, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses_list = []
    val_accuracies_list = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs} - Training:", end=" ")

        model.train()
        train_loss = 0.0
        train_acc = 0.0

        pbar = tqdm(train_loader, total=len(train_loader))
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc

            pbar.set_description(f"acc={train_acc/len(pbar):.1%}, loss={train_loss/len(pbar):.4f}")

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Store training metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)
                val_loss += loss.item()
                val_acc += acc

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Store validation metrics
        val_losses_list.append(val_loss)
        val_accuracies_list.append(val_acc)

        scheduler.step(val_loss)

        print(f"\nValidation Accuracy: {val_acc:.2%} | Validation Loss: {val_loss:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_wts)

    # Save model with class names
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': train_loader.dataset.classes
    }, save_path)

    print(f"Training complete. Best Validation Accuracy: {best_val_acc:.2%}")
    return model, train_losses, train_accuracies, val_losses_list, val_accuracies_list

# ========= Evaluation Function (New) =========
def evaluate_model(model, dataloader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate and print metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nFinal Test Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(xticks_rotation=45, cmap='Blues', ax=ax)
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2f})")
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.show()


# ========= Main Execution (Updated) =========
if __name__ == "__main__":
    # Load datasets inside main block (Windows multiprocessing fix)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}, Total: {num_classes}")

    model, train_losses, train_accuracies, val_losses_list, val_accuracies_list = train_model(
        create_model(num_classes), train_loader, val_loader, epochs=15)
        
    # Plotting Accuracy and Loss curves
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies_list) + 1), val_accuracies_list, label='Validation Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy vs. Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses_list) + 1), val_losses_list, label='Validation Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy_loss_graphs.png"))
    plt.show()
    
    # Evaluate on the validation set to get final metrics and confusion matrix
    evaluate_model(model, val_loader, train_dataset.classes)