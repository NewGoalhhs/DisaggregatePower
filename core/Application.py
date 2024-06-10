from helper.PrintHelper import PrintHelper
from screen.navigation.HomeScreen import HomeScreen


class Application:
    def __init__(self):
        pass

    def run(self):
        home_screen = HomeScreen()
        p = PrintHelper(home_screen=home_screen)
        home_screen.run(p=p)
