import os
from abc import ABC

from core.Screen import Screen
from features.predict.PredictModel import PredictModel


class PredictScreen(Screen):
    def screen(self, p):
        p.reset_lines()
        p.open_options()
        for option in self.get_generate_options(p):
            p.add_option(option['key'], option['text'], option['function'])
        p.choose_option()

    def get_generate_options(self, p) -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('MachineLearning/models') as entries:
            options = []
            for index, entry in enumerate(entries):
                if not entry.is_file():
                    continue

                model = entry.name.split('.')[0]

                instance = PredictModel(entry)
                options.append({
                    'key': str(index + 1),
                    'text': model,
                    'function': instance.predict
                })

            return options

    def simple_generate(self):
        pass