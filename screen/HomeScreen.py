from core.Screen import Screen
from screen.GenerateScreen import GenerateScreen
from screen.PredictScreen import PredictScreen
from screen.TrainScreen import TrainScreen


class HomeScreen(Screen):
    def screen(self, p):
        p.print_heading('Welcome to the application')

        p.print_line('Choose what you want to do:')
        p.open_options()
        p.add_option('1', 'Generate', GenerateScreen(self))
        p.add_option('2', 'Train', TrainScreen(self))
        p.add_option('3', 'Predict', PredictScreen(self))
        p.choose_option()
