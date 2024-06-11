from core.Screen import Screen
from screen.navigation.DocumentScreen import DocumentScreen
from screen.navigation.GenerateScreen import GenerateScreen
from screen.navigation.PredictScreen import PredictScreen
from screen.navigation.TrainScreen import TrainScreen


class HomeScreen(Screen):
    def screen(self, p):
        p.print_heading('Welcome to the application')

        p.print_line('Choose what you want to do:')
        p.open_options()
        p.add_option('1', 'Generate', GenerateScreen(self))
        p.add_option('2', 'Train', TrainScreen(self))
        p.add_option('3', 'Predict', PredictScreen(self))
        p.add_option('4', 'Document', DocumentScreen(self))
        p.choose_option()
