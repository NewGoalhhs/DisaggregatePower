import os

import keyboard
from core.Generate import Generate
from core.Screen import Screen


class PrintHelper:
    def __init__(self, primary_color: str = 'blue', secondary_color: str = 'green'):
        self.options = None
        self.primary_color = self.get_color_code(primary_color)
        self.secondary_color = self.get_color_code(secondary_color)
        self.lines = 0

    def print_heading(self, text: str):
        print()
        print(self.secondary_color + '-' * 50)
        print()
        print(' ' + self.primary_color + text)
        print()
        print(self.secondary_color + '-' * 50)
        print(self.get_color_code('reset'))
        self.lines += 7

    def print_line(self, text: str = ''):
        print(text)
        self.lines += 1

    def open_options(self):
        print()
        print(self.secondary_color + '-' * 50)
        print()
        self.options = {}
        self.lines += 3

    def add_option(self, key: str, text: str, function: callable):
        print(self.get_color_code('reset') + f" [" + self.secondary_color + f"{key}" + self.get_color_code('reset') + f"] " + self.primary_color + text)
        self.options[key] = {
            'text': text,
            'function': function
        }
        self.lines += 1

    def choose_option(self):
        print()
        print(self.secondary_color + '-' * 50)
        print(self.get_color_code('reset'))
        # result = self.request_input('Enter an option: ', autocomplete=list(self.options.keys()))
        result = self.request_input('Enter an option: ')

        if result in self.options:
            function = self.options[result]['function']
            if function is None:
                return
            elif function == 'exit':
                exit()
            elif isinstance(function, Screen):
                return function.screen(p=self)
            elif isinstance(function, Generate):
                return function.run(p=self)
            elif function is callable:
                return function(p=self)

        else:
            self.print_line('Invalid option')
            return self.choose_option()

    def reset_lines(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        self.lines = 0

    def request_input(self, text: str):
        return input(text)

    @staticmethod
    def get_color_code(color: str) -> str:
        color_codes = {
            'black': '\033[30m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'reset': '\033[0m'
        }
        return color_codes.get(color, '\033[0m')  # default to 'reset' if color not found

    def get_skip_keys(self):
        return ['up', 'down', 'left', 'right', 'shift', 'ctrl', 'alt', 'tab', 'esc', 'insert', 'delete', 'backspace',
                'enter', 'space', 'caps lock', 'num lock', 'scroll lock', 'print screen', 'pause', 'page up', 'home',
                'end']
