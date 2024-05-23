import math
import time

from helper.BasePrintHelper import BasePrintHelper
from helper.IntervalHelper import IntervalHelper


class LoadingBarHelper(BasePrintHelper):
    def __init__(self, text, goal, current=0, start=75, length=50, primary_color='blue', secondary_color='green', status='', tickrate=20):
        self.text = text
        self.goal = goal
        self.status = status
        self.tickrate = tickrate
        self.current = current
        self.start = start
        self.length = length
        self.previous_operations = 0
        self.start_time = time.time()
        self.primary_color = self.get_color_code(primary_color)
        self.secondary_color = self.get_color_code(secondary_color)
        self.interval = IntervalHelper(self.tickrate, self.print)

    def __str__(self):
        bar_length = int(self.current / self.goal * self.length) if self.goal != 0 else 0
        loading_bar_str = self.get_color_code('reset') + f"{self.text} - {self.status} "
        white_space_length = self.start - len(loading_bar_str)
        loading_bar_str += ' ' * white_space_length
        loading_bar_str += '[' + self.primary_color
        loading_bar_str += '|' * bar_length
        loading_bar_str += ' ' * (self.length - bar_length)
        loading_bar_str += self.get_color_code('reset') + ']'
        percentage = self.current/self.goal if self.goal != 0 else 0
        percentage = round(percentage * 100, 2) if percentage != 0 else 0
        loading_bar_str += f" {percentage}% {self.current:,}/{self.goal:,}"
        return loading_bar_str

    def set_goal(self, goal):
        self.goal = goal

    def set_text(self, text):
        self.text = text

    def set_status(self, status):
        self.status = status

    def set_current(self, current):
        self.current = current

    def update(self, current=1):
        self.current += current

    def predict_total_time(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            speed = self.current / elapsed_time
            if speed > 0:
                remaining_time = (self.goal - self.current) / speed
                return self.format_time(remaining_time)
        return 'Calculating...'

    def format_time(self, seconds):
        hours = math.floor(seconds / 3600)
        seconds = seconds % 3600
        minutes = math.floor(seconds / 60)
        seconds = round(seconds % 60, 2)
        hours_str = str(hours) + 'h ' if hours > 0 else ''
        minutes_str = str(minutes) + 'm ' if minutes > 0 else ''
        seconds_str = str(seconds) + 's' if seconds > 0 else ''
        return hours_str + minutes_str + seconds_str

    def print(self):
        print('\r' + self.__str__() + ' ' + self.predict_total_time(), end='')

    def finish(self):
        self.current = self.goal
        print('\r' + self.__str__() + self.secondary_color + ' DONE' + self.get_color_code('reset'))
        self.interval.cancel()
