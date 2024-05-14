from abc import abstractmethod


class Generate:
    def __init__(self):
        self.data = None

    @abstractmethod
    def run(self, p):
        pass