import os
from abc import ABC

from core.Screen import Screen
from features.predict.PredictModel import PredictModel


class PredictScreen(Screen):
    def screen(self, p):
        p.reset_lines()
        p.open_options()
        p.print_line("Which model do you want to use?")
        for option in self.get_generate_options(p):
            p.add_option(option['key'], option['text'], option['function'])
        p.choose_option()

    def get_generate_options(self, p) -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('MachineLearning/models') as entries:
            options = []
            for entry in entries:
                if entry.is_dir():
                    with os.scandir('MachineLearning/models/'+entry.name) as sub_entries:
                        for index, sub_entry in enumerate(sub_entries):
                            model = sub_entry.name.split('.')[0]

                            instance = PredictModel(entry.name + '/' + sub_entry.name)
                            options.append({
                                'key': str(index + 1),
                                'text': model,
                                'function': instance.predict
                            })

            return options

    def simple_generate(self):
        pass