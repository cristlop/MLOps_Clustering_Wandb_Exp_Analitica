import torch
import torchvision
from torch.utils.data import TensorDataset
from sklearn import datasets
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    # Load the Wisconsin Breast Cancer dataset
    wbcd = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names

    x, y = torch.tensor(wbcd.data).float(), torch.tensor(wbcd.target)

    # split off a validation set for hyperparameter tuning
    split_idx = int(len(x) * train_size)
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    training_set = TensorDataset(x_train, y_train)
    validation_set = TensorDataset(x_val, y_val)
    datasets = [training_set, validation_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-2024",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:

        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "wisconsin-breast-cancer-raw", type="dataset",
            description="raw Wisconsin Breast Cancer dataset, split into train/val",
            metadata={"source": "sklearn.datasets.load_breast_cancer",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
load_and_log()
