import os
from abc import ABC

from core.Screen import Screen


class GenerateScreen(Screen):
    def screen(self, p):
        p.reset_lines()
        p.open_options()
        for option in self.get_generate_options():
            p.add_option(option['key'], option['text'], option['function'])
        p.choose_option()

    def get_generate_options(self) -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('features/generate') as entries:
            options = []
            for index, entry in enumerate(entries):
                if not entry.is_file():
                    continue

                module = __import__('features.generate.' + entry.name.split('.')[0], fromlist=[entry.name.split('.')[0]])
                class_ = getattr(module, entry.name.split('.')[0])
                instance = class_()
                options.append({
                    'key': str(index + 1),
                    'text': entry.name,
                    'function': instance
                })

            return options

    def simple_generate(self):
        pass