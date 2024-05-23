from abc import abstractmethod


class MachineLearningModel:
    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def save_model(self, model):
        pass

    @abstractmethod
    def load_model(self, model):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass