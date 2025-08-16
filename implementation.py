#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path

# Paths relative to this file
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'model'
WEIGHTS_PATH = MODEL_DIR / 'best_model_weights.pth'

# Device-agnostic setup
def setup_device():
    """Setup device-agnostic code for PyTorch"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return device

# Dataset and Dataloader Setup
def get_dataloaders(batch_size=32):
    # Use consistent transforms for train and test
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    trainset = datasets.FashionMNIST(
        root='data',
        train=True,
        transform=transform,
        download=True
    )
    testset = datasets.FashionMNIST(
        root='data',
        train=False,
        transform=transform,
        download=True
    )
    train_dataloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=True
    )
    test_dataloader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        shuffle=False  # Don't shuffle test data
    )
    return trainset, testset, train_dataloader, test_dataloader

# Fixed Model Definition
class ImageClassifier(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_shape,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_shape,
                out_channels=hidden_shape,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# Fixed Training function with device support
def train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(train_dataloader):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training metrics
            train_loss += loss.item()
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            train_correct += (y_pred_class == y).sum().item()
            train_total += y.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_acc = (train_correct / train_total) if train_total else 0.0

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.inference_mode():
            for x, y in test_dataloader:
                # Move data to device
                x, y = x.to(device), y.to(device)
                
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                val_loss += loss.item()
                
                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
                val_correct += (y_pred_class == y).sum().item()
                val_total += y.size(0)

        # Calculate validation metrics
        avg_val_loss = val_loss / len(test_dataloader)
        val_acc = (val_correct / val_total) if val_total else 0.0

        # Print progress
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(WEIGHTS_PATH))
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.4f})")
        print("-" * 50)
    
    # Training summary
    print(f"\n🎉 Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {WEIGHTS_PATH}")

# Fixed Evaluation function with device support
def eval_mode(model, test_dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for _batch, (x, y) in enumerate(test_dataloader):
            # Move data to device
            x, y = x.to(device), y.to(device)
            
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            pred = torch.softmax(y_pred, dim=1).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    final_accuracy = (correct / total) if total else 0.0
    average_loss = total_loss / len(test_dataloader)
    return {"accuracy": final_accuracy, "average_loss": average_loss}


def main():
    # Setup device
    device = setup_device()
    
    # Setup
    trainset, testset, train_dataloader, test_dataloader = get_dataloaders(batch_size=64)
    num_classes = len(trainset.classes)

    # Print dataset information
    print(f"\n📊 Dataset Information:")
    print(f"  Training samples: {len(trainset)}")
    print(f"  Test samples: {len(testset)}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Classes: {trainset.classes}")
    print(f"  Batch size: 64")

    # Model, loss and optimizer
    torch.manual_seed(42)
    model = ImageClassifier(input_shape=1, hidden_shape=32, output_shape=num_classes)  # Larger hidden size
    
    # Print model information
    print(f"\n🤖 Model Information:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Move model to device
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Print training configuration
    print(f"\n⚙️ Training Configuration:")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: 0.001")
    print(f"  Loss function: CrossEntropyLoss")
    print(f"  Epochs: 20")
    print(f"  Device: {device}")

    # Create model directory (script-relative)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Starting Training...")
    print("=" * 50)

    # Train
    train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=20, device=device)

    print(f"\n🔍 Loading best model for final evaluation...")

    # Load best model for evaluation
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    
    # Evaluate
    evaluated = eval_mode(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, device=device)
    print(f"\n📈 Final Test Results:")
    print(f"  Accuracy: {evaluated['accuracy']:.4f}")
    print(f"  Loss: {evaluated['average_loss']:.4f}")

if __name__ == "__main__":
    main()