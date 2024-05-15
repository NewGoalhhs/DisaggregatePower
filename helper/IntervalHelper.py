import threading
import time


class IntervalHelper:
    def __init__(self, ms, func):
        self.ms = ms
        self.func = func
        self.stop = False
        self.timer = self.run()

    def run(self):
        if self.stop:
            self.func()
            return

        def func_wrapper():
            if self.stop:
                return
            self.run()
            self.func()

        timer = threading.Timer(interval=self.ms / 1000, function=func_wrapper)
        timer.start()
        return timer

    def cancel(self):
        self.stop = True
