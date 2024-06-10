from abc import abstractmethod


class MachineLearningModel:

    def __init__(self, print_progress):
        self.print_progress = print_progress
        pass
    @abstractmethod
    def preprocess_data(self, data) -> tuple:
        pass

    def save_model(self, model):
        pass

    def load_model(self, model):
        pass

    @abstractmethod
    def train(self, data, epochs=100, print_progress=True):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def get_score(self, y, y_pred) -> float:
        pass

    @abstractmethod
    def file_extension(self):
        pass

    def get_document_parameters(self) -> dict:
        return {}
