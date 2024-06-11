from abc import ABC

from core.Database import Database
from core.Screen import Screen
from SQL.SQLQueries import DatabaseOperations as Query


class AbstractModelScreen(Screen, ABC):
    @classmethod
    def get_appliance(cls, p):
        appliances = Database.query(Query.SELECT_ALL.format('Appliance'))
        p.print_line("All appliances:")
        p.open_options()
        for index, appliance in enumerate(appliances):
            p.add_option(index + 1, appliance['name'], appliance)
        return p.choose_option('Choose an appliance to train: ')

    @classmethod
    def get_epochs(cls, p):
        result = p.request_input("Enter the number of epochs to train [100]: ", condition=lambda x: x.isdigit(), default='100')
        return int(result)