from abc import abstractmethod


class Screen:
    def __init__(self, previous_screen=None):
        self.previous_screen = previous_screen
        pass

    def run(self, p):
        p.set_previous_screen(self.previous_screen)
        self.screen(p=p)

    @abstractmethod
    def screen(self, p):
        pass
