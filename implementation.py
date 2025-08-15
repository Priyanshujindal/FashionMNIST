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

# Fixed Training function
def train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()

        for _batch, (x, y) in enumerate(train_dataloader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for x, y in test_dataloader:
                y_pred = model(x)
                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
                val_correct += (y_pred_class == y).sum().item()
                val_total += y.size(0)

        # Calculate metrics needed for model selection
        val_acc = (val_correct / val_total) if val_total else 0.0

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(WEIGHTS_PATH))

# Fixed Evaluation function
def eval_mode(model, test_dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for _batch, (x, y) in enumerate(test_dataloader):
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
    # Setup
    trainset, testset, train_dataloader, test_dataloader = get_dataloaders(batch_size=64)
    num_classes = len(trainset.classes)

    # Model, loss and optimizer
    torch.manual_seed(42)
    model = ImageClassifier(input_shape=1, hidden_shape=32, output_shape=num_classes)  # Larger hidden size
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create model directory (script-relative)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs=20)

    # Load best model for evaluation
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location='cpu'))
    
    # Evaluate
    evaluated = eval_mode(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn)
    print(f"Accuracy: {evaluated['accuracy']:.4f} | Loss: {evaluated['average_loss']:.4f}")

if __name__ == "__main__":
    main()