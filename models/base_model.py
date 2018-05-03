from abc import ABCMeta, abstractmethod
import numpy as np

class Model(metaclass=ABCMeta):
    """Abstract model class"""

    @abstractmethod
    def initialize_params(self, abstracts, labels):
        """
        Initializes parameters used by the model based on predefined settings
        """

        raise NotImplementedError

    @abstractmethod
    def train(self, abstracts, parameters):
        """
        Trains the model using the parameters created using initialize_params
        """

        raise NotImplementedError

    def run_model(self, abstracts, labels):
        parameters = self.initialize_params(abstracts, labels)
        model = self.train(abstracts, parameters)

        return model