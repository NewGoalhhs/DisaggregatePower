import os
from abc import ABC

from core.Database import Database
from core.Screen import Screen
from screen.operation.PredictModelScreen import PredictModelScreen
from SQL.SQLQueries import DatabaseOperations as Query


class PredictScreen(Screen):
    def screen(self, p):
        self.p = p
        p.reset_lines()
        p.open_options()
        p.print_line("Which model do you want to use?")
        for option in self.get_generate_options(p):
            p.add_option(option['key'], option['text'], option['function'], option['args'])
        p.choose_option()

    def choose_option(self, args):
        entry, sub_entry = args
        appliance = self.get_appliance(sub_entry.name)
        instance = PredictModelScreen(entry.name + '/' + sub_entry.name, appliance)
        instance.screen(p=self.p)

    def get_generate_options(self, p) -> list:
        # Return a list of indexes and generate class options from features/generate
        with os.scandir('MachineLearning/models') as entries:
            options = []
            for entry in entries:
                if entry.is_dir():
                    with os.scandir('MachineLearning/models/'+entry.name) as sub_entries:
                        for index, sub_entry in enumerate(sub_entries):
                            model = sub_entry.name.split('.')[0]

                            options.append({
                                'key': str(index + 1),
                                'text': model,
                                'function': self.choose_option,
                                'args': [entry, sub_entry]
                            })

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
