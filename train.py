from processors.preprocessor import Imputer, Normalizer
from processors.processor import Processor
import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils.utils import *
import time
from pathlib import Path
from models.neural_net import NeuralNet

test_number = 1
save_path = "./tmp/" + str(test_number)
Config = load_config("config.yaml", save_path)

X, Y = load_dataset(year=1, shuffle=True)

imputer = Imputer(strategy='mean')
normalizer = Normalizer(strategy="l2", norm_axis=1)
processor = Processor(X, Y, 15, 10, "regularize", 64, True)

X = imputer.fit_transform(abstracts=X)
X = normalizer.transform(abstracts=X)

# x_train, y_train, x_dev, y_dev, x_test, y_test = processor.split_data()
# print(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape, y_test.shape)

neural_net = NeuralNet(**Config["HyperParameters"], **Config["Processor"], **Config["Progress"])

neural_net.initialize_params(X, Y, [0, 1])

