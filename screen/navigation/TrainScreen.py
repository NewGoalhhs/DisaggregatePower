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
        for option in self.get_generate_options():
            p.add_option(option['key'], option['text'], option['function'], option['args'])
        p.choose_option()

    def choose_option(self, model_name):
        module = __import__('MachineLearning.' + model_name, fromlist=[model_name])
        model = getattr(module, model_name)
        instance = TrainModelScreen(model)
        instance.screen(p=self.p)

    def get_generate_options(self) -> list:
        with os.scandir(__ROOT__+'/MachineLearning') as entries:
            options = []
            index = 0
            for entry in entries:
                if entry.is_dir():
                    continue

                name = entry.name.split('.')[0]

                options.append({
                    'key': str(index + 1),
                    'text': name,
                    'function': self.choose_option,
                    'args': name
                })

                index += 1

            return options

    def simple_generate(self):
        pass