import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import wandb

# Ignorar las advertencias de convergencia
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Instalar WandB si no está instalado
!pip install wandb -qU

# Iniciar sesión en WandB
wandb.login()

# Cargar datos de vivienda de California
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X, y = X[::2], y[::2]  # Submuestrear para una demostración más rápida

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Inicializar la ejecución de WandB para la regresión
run = wandb.init(project='my-scikit-integration', name="regression")

# Crear un modelo de regresión Ridge
reg = Ridge()
reg.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = reg.predict(X_test)

# Calcular la métrica de rendimiento (por ejemplo, error cuadrático medio en este caso)
mse = mean_squared_error(y_test, y_pred)

# Registrar el modelo y la métrica en WandB
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')
wandb.log({"regression/mse": mse})

# Finalizar la ejecución de WandB
wandb.finish()
