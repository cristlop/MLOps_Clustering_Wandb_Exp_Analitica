import torch
import torchvision
from torch.utils.data import TensorDataset
import argparse
import wandb
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
 
    # the data, split between train and test sets
    # Use iris dataset instead of MNIST
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # split off a validation set for hyperparameter tuning
    x_train, x_val = X[:int(len(X)*train_size)], X[int(len(X)*train_size):]
    y_train, y_val = y[:int(len(X)*train_size)], y[int(len(X)*train_size):]

    training_set = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    validation_set = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    datasets = [training_set, validation_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-Clustering-2024",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "iris-raw", type="dataset",
            description="raw Iris dataset, split into train/val",
            metadata={"source": "sklearn.datasets.load_iris",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

        # Now integrate clustering with W&B
        # Train model
        kmeans = KMeans(n_clusters=4, random_state=1)
        cluster_labels = kmeans.fit_predict(datasets[0].tensors[0])

        # Visualize model performance
        wandb.sklearn.plot_elbow_curve(kmeans, datasets[0].tensors[0])
        wandb.sklearn.plot_silhouette(kmeans, datasets[0].tensors[0], cluster_labels)

        # All in one: Clusterer Plot
        wandb.sklearn.plot_clusterer(kmeans, datasets[0].tensors[0], cluster_labels, datasets[0].tensors[1], 'KMeans')

        wandb.finish()

# Call the function to load and log data
load_and_log()
