from helper.PrintHelper import PrintHelper
from screen.HomeScreen import HomeScreen

class Application:
    def __init__(self):
        pass

    def run(self):
        p = PrintHelper()
        HomeScreen().screen(p)