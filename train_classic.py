import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import urllib.request
import zipfile
import pickle

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import PrecisionRecallDisplay


def main():
    
    data_dir_drink = "yolo_dataset2/face"

    # Set up transformations for training and validation (test) data
    # For training data we will do randomized cropping to get to 224 * 224,
    # randomized horizontal flipping, and normalization.
    # For test set we will do only center cropping to get to 224 * 224,
    # and normalization.

    data_transforms = {
      'train': transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485], [0.229])
      ]),
      'val': transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.Resize(256),
          transforms.CenterCrop(224),
          transforms.ToTensor(),
          transforms.Normalize([0.485], [0.229])
      ]),
    }

    # Create Datasets for training and validation sets
    train_dataset_drink = datasets.ImageFolder(
        os.path.join(data_dir_drink, 'train'),
        data_transforms['train'])
    val_dataset_drink = datasets.ImageFolder(
        os.path.join(data_dir_drink, 'val'),
        data_transforms['val'])

    # Create DataLoaders for training and validation sets
    batch_size = len(train_dataset_drink)
    train_loader_drink = torch.utils.data.DataLoader(
        train_dataset_drink, batch_size=batch_size,
        shuffle=True, num_workers=4)
    val_loader_drink = torch.utils.data.DataLoader(
        val_dataset_drink, batch_size=batch_size,
        shuffle=False, num_workers=4)

    # Set up dict for dataloaders
    dataloaders = {'train':train_loader_drink,'val':val_loader_drink}
    # Store size of training and validation sets
    dataset_sizes = {'train':len(train_dataset_drink),'val':len(val_dataset_drink)}
    # Get class names associated with labels
    classes_drink = train_dataset_drink.classes

    # Define features and target
    images_drink, labels_drink = iter(train_loader_drink).next()
    test_images_drink, test_labels_drink = iter(val_loader_drink).next()

    # Convert to numpy arrays to train the LR model
    X_drink = torch.flatten(images_drink, start_dim=1)
    X_drink = X_drink.numpy()
    y_drink = labels_drink.numpy()

    # Test set
    X_test_drink = torch.flatten(test_images_drink, start_dim=1)
    X_test_drink = X_test_drink.numpy()
    y_test_drink = test_labels_drink.numpy()

    # Fit model to the data
    clf = LogisticRegression(penalty='none', 
                          tol=0.1, solver='saga',
                          multi_class='multinomial').fit(X_drink, y_drink)

    # Generage preds
    z = [ clf.intercept_[k] + np.dot(clf.coef_[k], X_test_drink[k]) for k in range(1) ]

    #conditional probability
    exps = [np.exp(z[k]) for k in range(1)]
    exps_sum = np.sum(exps)
    probs = exps/exps_sum

    #predictied label
    idx_cls = np.argmax(probs)

    #prediction probabilities
    y_pred_prob_drink = clf.predict_proba(X_test_drink)

    #predicted label
    y_pred_drink = clf.predict(X_test_drink)

    # Confusion Matrix
    cm = pd.crosstab(y_test_drink, y_pred_drink, 
                                rownames=['Actual'], colnames=['Predicted'], normalize='index')
    p = plt.figure(figsize=(10,10));
    p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)

    # Acccuracy
    return print("Train Accuracy :", clf.score(X_drink, y_drink)), print("Test Accuracy :", clf.score(X_test_drink, y_test_drink)), p


if __name__ == '__main__':
    main()