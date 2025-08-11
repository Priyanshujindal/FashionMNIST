#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import Accuracy
from timeit import default_timer as Timer
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# Dataset and Dataloader Setup
def get_dataloaders(batch_size=32):
    trainset = datasets.FashionMNIST(
        root='data',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    testset = datasets.FashionMNIST(
        root='data',
        train=False,
        transform=transforms.ToTensor(),
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
        shuffle=True
    )
    return trainset, testset, train_dataloader, test_dataloader

# Model Definition
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_shape, hidden_shape, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_shape * 7 * 7, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ans = self.block_1(x)
        ans = self.block_2(ans)
        ans = self.classifier(ans)
        return ans

# Training function
def train_model(model, train_dataloader, loss_fn, optimizer, epochs, num_classes):
    import tqdm
    accuracy = Accuracy(task='multiclass', num_classes=num_classes)
    model.train()
    for epoch in tqdm.tqdm(range(epochs), desc='Training'):
        total_loss = 0.0
        accuracy.reset()
        for batch, (x, y) in enumerate(train_dataloader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            y_pred_class = torch.softmax(y_pred, dim=1).argmax(dim=1)
            acc = accuracy(y_pred_class, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss = total_loss / len(train_dataloader)
        accu = accuracy.compute()
        tqdm.tqdm.write(f"Epoch: {epoch} | Average Loss: {average_loss:.4f} | Accuracy: {accu:.4f}")

# Evaluation function
def eval_mode(model, test_dataloader, loss_fn, num_class=9):
    model.eval()
    total_loss = 0.0
    test_accuracy = Accuracy(task='multiclass', num_classes=num_class)
    start_time = Timer()
    all_pred = []
    all_target = []
    with torch.inference_mode():
        for batch, (x, y) in enumerate(test_dataloader):
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            pred = torch.softmax(y_pred, dim=1).argmax(dim=1)
            acc = test_accuracy(pred, y)
            all_pred.append(pred)
            all_target.append(y)
            if batch % 100 == 0:
                print(f"Batch: {batch} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
    final_accuracy = test_accuracy.compute()
    average_loss = total_loss / len(test_dataloader)
    eval_time = Timer() - start_time
    y_pred_tensor = torch.cat(all_pred, dim=0)
    y_pred_target = torch.cat(all_target, dim=0)
    print(f"Final Accuracy: {final_accuracy:.4f}")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    return {
        "accuracy": final_accuracy,
        "average_loss": average_loss,
        "eval_time": eval_time,
        "y_pred_tensor": y_pred_tensor,
        "y_target": y_pred_target
    }

# Optional: visualize some training images (call this explicitly if needed)
def visualize_samples(trainset, classes):
    import matplotlib.pyplot as plt
    torch.manual_seed(42)
    fig = plt.figure(figsize=(9, 9))
    row, col = 4, 4
    for i in range(1, row * col + 1):
        index = torch.randint(0, len(trainset), size=[1]).item()
        image, label = trainset[index]
        fig.add_subplot(row, col, i)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(classes[label])
        plt.axis('off')
    plt.show()

# Function to plot confusion matrix given eval results
def plot_confusion(eval_results, classes):
    confmat = ConfusionMatrix(num_classes=len(classes), task='multiclass')
    confmat_tensor = confmat(preds=eval_results["y_pred_tensor"], target=eval_results["y_target"])
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=classes,
        figsize=(10, 7)
    )
    plt.show()

# Main function to execute training, evaluation, and saving
def main():
    # Setup
    trainset, testset, train_dataloader, test_dataloader = get_dataloaders(batch_size=32)
    classes = trainset.classes
    num_classes = len(classes)

    # Optional: visualize some samples if needed
    # visualize_samples(trainset, classes)

    # Model, loss and optimizer
    torch.manual_seed(42)
    model = ImageClassifier(input_shape=1, hidden_shape=10, output_shape=num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    # Train
    train_model(model, train_dataloader, loss_fn, optimizer, epochs=10, num_classes=num_classes)

    # Evaluate
    evaluated = eval_mode(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, num_class=num_classes)

    # Plot confusion matrix
    plot_confusion(evaluated, classes)

    # Save model weights
    model_path = "model"
    os.makedirs(model_path, exist_ok=True)
    model_name = os.path.join(model_path, 'model_weights.pth')
    torch.save(model.state_dict(), model_name)
    print(f"Model weights saved to {model_name}")

# This guard ensures no code runs on import, only if run directly
if __name__ == "__main__":
    main()
