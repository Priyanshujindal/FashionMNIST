#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn

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
def train_model(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, log_every=None, print_on_improve=False):
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for _batch, (x, y) in enumerate(train_dataloader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            train_correct += (y_pred_class == y).sum().item()
            train_total += y.size(0)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():
            for x, y in test_dataloader:
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                total_val_loss += loss.item()

                y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
                val_correct += (y_pred_class == y).sum().item()
                val_total += y.size(0)

        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(test_dataloader)
        train_acc = (train_correct / train_total) if train_total else 0.0
        val_acc = (val_correct / val_total) if val_total else 0.0

        # Save best model and optionally print on improvement
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'model/best_model_weights.pth')
            if print_on_improve:
                print(
                    f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} (improved)"
                )

        # Optionally print at interval
        if log_every is not None and log_every > 0 and ((epoch + 1) % log_every == 0):
            print(
                f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    print(f"Best validation accuracy: {best_val_acc:.4f}")

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
 
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Average Loss: {average_loss:.4f}")

    return {
        "accuracy": final_accuracy,
        "average_loss": average_loss,
    }

 

def main():
    # Setup
    trainset, testset, train_dataloader, test_dataloader = get_dataloaders(batch_size=64)
    classes = trainset.classes
    num_classes = len(classes)
    
    # Minimal metadata output
    print(f"Classes: {classes} | Train: {len(trainset)} | Test: {len(testset)}")

    # Model, loss and optimizer
    torch.manual_seed(42)
    model = ImageClassifier(input_shape=1, hidden_shape=32, output_shape=num_classes)  # Larger hidden size
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create model directory
    os.makedirs('model', exist_ok=True)

    # Train
    train_model(
        model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        epochs=20,
        log_every=None,
        print_on_improve=True,
    )

    # Load best model for evaluation
    model.load_state_dict(torch.load('model/best_model_weights.pth'))
    
    # Evaluate
    evaluated = eval_mode(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn)

    # Confusion matrix plotting removed for a leaner dependency footprint
    print(f"Accuracy: {evaluated['accuracy']:.4f} ({evaluated['accuracy']*100:.2f}%) | Loss: {evaluated['average_loss']:.4f}")

if __name__ == "__main__":
    main()