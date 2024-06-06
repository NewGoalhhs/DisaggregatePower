import os

from core.Screen import Screen
from features.train.TrainModel import TrainModel


class TrainScreen(Screen):
    def screen(self, p):
        p.reset_lines()
        p.open_options()
        for option in self.get_generate_options():
            p.add_option(option['key'], option['text'], option['function'])
        p.choose_option()

    def get_generate_options(self) -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('MachineLearning') as entries:
            options = []
            index = 0
            for entry in entries:
                if not entry.is_file():
                    continue


                name = entry.name.split('.')[0]
                module = __import__('MachineLearning.' + name,
                                    fromlist=[name])
                class_ = getattr(module, name)
                model_instance = class_()

                instance = TrainModel(model_instance)
                options.append({
                    'key': str(index + 1),
                    'text': name,
                    'function': instance.train
                })
                
                index += 1

            return options

    def simple_generate(self):
        pass