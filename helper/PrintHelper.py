import inspect
import os

from core.Generate import Generate
from core.Screen import Screen
from helper.BasePrintHelper import BasePrintHelper
from helper.LoadingBarHelper import LoadingBarHelper


class PrintHelper(BasePrintHelper):
    def __init__(self, primary_color: str = 'blue', secondary_color: str = 'green'):
        self.options = None
        self.primary_color = self.get_color_code(primary_color)
        self.secondary_color = self.get_color_code(secondary_color)

    def print_heading(self, text: str):
        print()
        print(self.secondary_color + '-' * 50)
        print()
        print(' ' + self.primary_color + text)
        print()
        print(self.secondary_color + '-' * 50)
        print(self.get_color_code('reset'))

    def print_line(self, text: str = ''):
        print(text)

    def open_options(self):
        print()
        print(self.secondary_color + '-' * 50)
        print()
        self.options = {}

    def add_option(self, key: str, text: str, function: callable):
        print(self.get_color_code('reset') + f" [" + self.secondary_color + f"{key}" + self.get_color_code(
            'reset') + f"] " + self.primary_color + text)
        self.options[str(key)] = {
            'text': text,
            'function': function
        }

    def choose_option(self, text: str = 'Enter an option: '):
        print()
        print(self.secondary_color + '-' * 50)
        print(self.get_color_code('reset'))
        # result = self.request_input('Enter an option: ', autocomplete=list(self.options.keys()))
        result = self.request_input(text)

        if result in self.options.keys():
            function = self.options[result]['function']
            if function is None:
                return
            elif isinstance(function, Screen):
                return function.screen(p=self)
            elif isinstance(function, Generate):
                return function.run(p=self)
            elif (function is callable or
                  function is classmethod or
                  function is staticmethod or
                  inspect.ismethod(function)):
                return function(p=self)
            else:
                return function

        else:
            self.print_line('Invalid option')
            return self.choose_option()

    def reset_lines(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def request_input(self, text: str):
        return input(text)

    def get_loading_bar(self, text, goal, length=50) -> LoadingBarHelper:
        helper = LoadingBarHelper(text=text, goal=goal, length=length)
        helper.primary_color = self.primary_color
        helper.secondary_color = self.secondary_color
        helper.print()
        return helper
