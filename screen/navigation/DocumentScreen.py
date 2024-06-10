import os
from abc import ABC

from core.Database import Database
from core.Screen import Screen
from SQL.SQLQueries import DatabaseOperations as Query
from screen.operation.DocumentModelScreen import DocumentModelScreen


class DocumentScreen(Screen):
    def screen(self, p):
        p.reset_lines()
        p.open_options()
        p.print_line("Which model do you want to use?")
        for option in self.get_generate_options(p):
            p.add_option(option['key'], option['text'], option['function'])
        p.choose_option()

    def get_generate_options(self, p) -> list:
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

                instance = DocumentModelScreen(model_instance)
                options.append({
                    'key': str(index + 1),
                    'text': name,
                    'function': instance
                })

                index += 1

            return options

    def simple_generate(self):
        pass

    def get_appliance(self, file_name, additional=''):
        if not file_name:
            return ''

        split_entry = file_name.split('.')[0]
        appliance_name = split_entry.split('_')[-1] + additional
        appliance = Database.query(Query.SELECT_WHERE.format('Appliance', 'name', appliance_name))
        if len(appliance) > 0:
            return appliance[0]
        else:
            handling = split_entry.replace('_' + appliance_name, '')
            return self.get_appliance(handling, '_' + appliance_name)
