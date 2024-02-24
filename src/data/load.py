import torch
import torchvision
from torch.utils.data import TensorDataset
import argparse
import wandb
from sklearn import datasets
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    # Load the data
    """
      
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # split off a validation set for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=42)

    training_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    validation_set = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
    datasets = [training_set, validation_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-Pycon2023",
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
                X, y = data.tensors
                torch.save((X, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
