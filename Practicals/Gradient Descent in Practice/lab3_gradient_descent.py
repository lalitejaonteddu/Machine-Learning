#!/usr/bin/env python3
"""
Lab 3: Gradient Descent in Practice

This script trains a binary logistic regression model on the student placement dataset
using two optimization methods:

1) Batch Gradient Descent
2) Mini-batch Gradient Descent

How to run:
    python lab3_gradient_descent.py --mode batch
    python lab3_gradient_descent.py --mode mini-batch --batch-size 64
    python lab3_gradient_descent.py --mode batch --epochs 200 --lr 0.05
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).with_name("placement_predict_50k.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)

    # Use only numeric/meaningful features for a simple linear model.
    feature_cols = [
        "CGPA",
        "Internships",
        "Projects",
        "AptitudeScore",
        "SoftSkillsRating",
        "Backlogs",
        "PlacementTraining",
    ]

    # PlacementTraining is already a yes/no field, convert to binary.
    df["PlacementTraining"] = (df["PlacementTraining"] == "Yes").astype(float)

    X = df[feature_cols].to_numpy(dtype=float)
    y = (df["PlacementStatus"] == "Placed").astype(float).to_numpy(dtype=float)
    return X, y


def standardize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std
    return Xn, mean, std


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def compute_loss(X, y, w, b):
    z = X @ w + b
    p = sigmoid(z)
    eps = 1e-8
    loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)).mean()
    return float(loss)


def batch_gradient_descent(X, y, lr=0.05, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    losses = []
    for epoch in range(1, epochs + 1):
        z = X @ w + b
        p = sigmoid(z)

        # Logistic regression gradients
        dw = (1 / n_samples) * (X.T @ (p - y))
        db = (1 / n_samples) * np.sum(p - y)

        w -= lr * dw
        b -= lr * db

        loss = compute_loss(X, y, w, b)
        losses.append(loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss = {loss:.6f}")

    return w, b, losses


def mini_batch_gradient_descent(X, y, lr=0.05, epochs=200, batch_size=32):
    n_samples, n_features = X.shape
    w = np.zeros(n_features, dtype=float)
    b = 0.0

    losses = []
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            z = X_batch @ w + b
            p = sigmoid(z)

            dw = (1 / len(X_batch)) * (X_batch.T @ (p - y_batch))
            db = (1 / len(X_batch)) * np.sum(p - y_batch)

            w -= lr * dw
            b -= lr * db

        loss = compute_loss(X, y, w, b)
        losses.append(loss)

        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss = {loss:.6f}")

    return w, b, losses


def accuracy(X, y, w, b):
    probs = sigmoid(X @ w + b)
    preds = (probs >= 0.5).astype(float)
    return float(np.mean(preds == y))


def plot_losses(losses, title, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, color="royalblue", linewidth=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression with batch or mini-batch gradient descent")
    parser.add_argument("--mode", type=str, default="batch", choices=["batch", "mini-batch"], help="Optimization method")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size when using mini-batch mode")
    parser.add_argument("--plot", action="store_true", help="Display a loss-curve plot after training")
    parser.add_argument("--save-plot", type=str, default=None, help="Optional file path to save the loss plot")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y = load_data()
    X, mean, std = standardize(X)

    print("Dataset shape:", X.shape)
    print("Features used:", [
        "CGPA",
        "Internships",
        "Projects",
        "AptitudeScore",
        "SoftSkillsRating",
        "Backlogs",
        "PlacementTraining",
    ])
    print("Target rate:", y.mean())

    if args.mode == "batch":
        print("\nRunning Batch Gradient Descent...")
        w, b, losses = batch_gradient_descent(X, y, lr=args.lr, epochs=args.epochs)
        title = "Batch Gradient Descent Loss"
    else:
        print("\nRunning Mini-batch Gradient Descent...")
        w, b, losses = mini_batch_gradient_descent(X, y, lr=args.lr, epochs=args.epochs, batch_size=args.batch_size)
        title = "Mini-batch Gradient Descent Loss"

    train_acc = accuracy(X, y, w, b)
    final_loss = losses[-1]

    print("\nTraining summary")
    print("-" * 40)
    print(f"Final loss      : {final_loss:.6f}")
    print(f"Train accuracy  : {train_acc:.4f}")
    print(f"Weights         : {np.round(w, 4)}")
    print(f"Bias            : {b:.6f}")

    if args.plot:
        plot_losses(losses, title, save_path=args.save_plot)


if __name__ == "__main__":
    main()
