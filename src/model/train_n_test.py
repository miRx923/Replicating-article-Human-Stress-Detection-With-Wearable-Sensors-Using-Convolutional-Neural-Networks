import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score

from model.cnn_model import StressCNN
from dataset_class.stress_dataset import StressDataset
from utils import modify_labels, compute_ci



# Hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_and_test(data_path, task):
    """
    Function for training and testing a model on one of the three classification tasks
    """

    data = pd.read_pickle(data_path)
    X = data['X']
    y = data['y']
    sid = data['sid']

    y = modify_labels(y, task)
    num_classes = len(np.unique(y))

    # LOSO Cross-validation
    accuracies = []
    f1_scores = []
    ci_accuracies = []
    ci_f1_scores = []

    for test_subject in np.unique(sid):
        #print(f"Testing on subject {test_subject}")

        # Split dataset
        train_val_indices = sid != test_subject
        test_indices = sid == test_subject

        X_train_val, X_test = X[train_val_indices], X[test_indices]
        y_train_val, y_test = y[train_val_indices], y[test_indices]

        validation_subject = np.unique(sid[train_val_indices])[0]  # First subject for validation
        train_indices = sid[train_val_indices] != validation_subject
        val_indices = sid[train_val_indices] == validation_subject

        X_train, X_val = X_train_val[train_indices], X_train_val[val_indices]
        y_train, y_val = y_train_val[train_indices], y_train_val[val_indices]

        train_dataset = StressDataset(X_train, y_train)
        val_dataset = StressDataset(X_val, y_val)
        test_dataset = StressDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = StressCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

        # Training with validation
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)  # Add channel dimension -> (batch_size, 1, height, width)

                inputs = inputs.float().to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation after each epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if len(inputs.shape) == 3:
                        inputs = inputs.unsqueeze(1)
                    inputs = inputs.float().to(device)
                    labels = labels.long().to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            #print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()

        model.load_state_dict(best_model)
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                if len(inputs.shape) == 3:
                    inputs = inputs.unsqueeze(1)
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds) * 100
        test_f1 = f1_score(all_labels, all_preds, average='macro') * 100

        # (N in CI)
        n_samples = len(all_labels)

        # Calculate 95% CI for Accuracy and F1-score
        ci_acc = compute_ci(test_acc, n_samples)
        ci_f1 = compute_ci(test_f1, n_samples)

        accuracies.append(test_acc)
        f1_scores.append(test_f1)
        ci_accuracies.append(ci_acc)
        ci_f1_scores.append(ci_f1)

        #print(f"Subject {test_subject} - Accuracy: {test_acc:.3f}% ± {ci_acc:.3f}, F1 Score: {test_f1:.3f}% ± {ci_f1:.3f}")

    # Average metrics over all subjects - for the paper table 1.
    avg_accuracy = np.mean(accuracies)
    avg_f1_score = np.mean(f1_scores)
    avg_ci_acc = np.mean(ci_accuracies)
    avg_ci_f1 = np.mean(ci_f1_scores)

    print(f"Task: {task}, Data: {data_path}")
    print(f"Average Accuracy: {avg_accuracy:.3f}% ± {avg_ci_acc:.3f}")
    print(f"Average F1 Score: {avg_f1_score:.3f}% ± {avg_ci_f1:.3f}")

    return avg_accuracy, avg_f1_score, avg_ci_acc, avg_ci_f1



def experiments(data_files, tasks):
    """
    Function to run all experiments from the paper
    """

    for data_file in data_files:
        for task in tasks:
            train_and_test(data_file, task)
