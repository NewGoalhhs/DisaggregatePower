from core.Screen import Screen
from screen.GenerateScreen import GenerateScreen


class HomeScreen(Screen):
    def screen(self, p):
        p.print_heading('Welcome to the application')

        p.print_line('Choose what you want to do:')
        p.open_options()
        p.add_option('1', 'Generate', GenerateScreen())
        p.choose_option()
