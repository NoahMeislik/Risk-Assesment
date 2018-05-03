from abc import ABCMeta, abstractmethod
import numpy as np
import sklearn.preprocessing

class PreProcessor(metaclass=ABCMeta):
    """Abstract PreProcessor class"""

    @abstractmethod
    def fit(self, abstracts):
        """Fits the PreProcessor values to the abstracts"""
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, abstracts):
        """Uses the PreProcessor to transform the data"""
        raise NotImplementedError

    def fit_transform(self, abstracts):
        """Fits and then transforms abstracts"""
        self.fit(abstracts)
        return self.transform(abstracts)


class Imputer(PreProcessor):
    def __init__(self, strategy):
        """Initialize Object

        Required Args: 
            Self: Self variable from class
            Strategy: The strategy used to impute the NaN values in the abstracts
            Abstracts: The dataset used to impute on using the strategy mentioned above 
        Output:
            Imputed Abstracts: The imputed abstracts using the Strategy are returned
        """

        self.strategy = strategy

        if (self.strategy != 'mean' or self.strategy != 'median' or self.strategy != 'most_frequent'):
            #raise ValueError('Strategy specified must be of type mean, median or most_frequent!')
            pass

    def fit(self, abstracts):
        """Fits the abstracts to the imputer model"""

        self.imputer = sklearn.preprocessing.Imputer(missing_values = 0, strategy = self.strategy, axis=1)
        self.imputer.fit(abstracts, y=None)

    def transform(self, abstracts):
        """Transforms the abstracts to the imputer model using specified parameters"""
        # Sets all NaN to 0 in the dataset
        abstracts[np.isnan(abstracts)]=0

        abstracts = self.imputer.transform(abstracts)

        return abstracts

class Normalizer(PreProcessor):
    def __init__(self, strategy, norm_axis = 1):
        """Initialize Object

        Required Args:
            Self: Self variable from class
            Strategy: The strategy used to normalize the data (l1, l2, or max)
            Abstracts: The dataset used to normalize on using the strategy mentioned above 
        Output:
            Normalized Abstracts: The normalized abstracts using the Strategy are returned
        """

        self.strategy = strategy
        self.norm_axis = norm_axis

        if (self.strategy != 'l1' or self.strategy != 'l2' or self.strategy != 'max'):
            #raise ValueError('Strategy specified must be of type l1, l2 or max!')
            pass

        if (self.norm_axis != 0 or self.norm_axis != 1):
            #raise ValueError('Normalization axis must be of value 0 or 1')
            pass

    def fit(self, abstracts):
        """Fits the abstracts to the normalization model"""

        pass

    def transform(self, abstracts):
        """Transforms the abstracts to the normalization model using specified parameters"""

        abstracts = sklearn.preprocessing.normalize(abstracts, norm = self.strategy, axis = self.norm_axis)

        return abstracts

            


        

