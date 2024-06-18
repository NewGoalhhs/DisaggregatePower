import os

from core.Screen import Screen
from features.train.TrainModel import TrainModel
from screen.operation.TrainModelScreen import TrainModelScreen
from app import __ROOT__


class TrainScreen(Screen):
    def screen(self, p):
        self.p = p
        p.reset_lines()
        p.open_options()
        for index, option in enumerate(TrainModel.get_model_options()):
            p.add_option(
                str(index+1),
                option,
                lambda: self.choose_option(option),
                option
            )
        p.choose_option()

    def choose_option(self, model_name):
        module = __import__('MachineLearning.' + model_name, fromlist=[model_name])
        model = getattr(module, model_name)
        instance = TrainModelScreen(model)
        instance.screen(p=self.p)

    def simple_generate(self):
        pass