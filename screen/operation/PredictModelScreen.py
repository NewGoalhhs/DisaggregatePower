import datetime

from core.Database import Database
from core.Screen import Screen
from features.document.DocumentModel import DocumentModel
from features.predict.PredictModel import PredictModel
from SQL.SQLQueries import DatabaseOperations as Query


class PredictModelScreen(Screen):
    def __init__(self, model_path, appliance):
        super().__init__()

        self.predict_model = PredictModel(model_path, appliance)
        self.document_model = DocumentModel(self.predict_model.model)

    def screen(self, p):
        p.print_heading('Predict Model')

        p.print_line('How do you want to predict?')
        p.open_options()
        p.add_option('1', 'Gather interesting datetime from the database (lasy method)', '1')
        p.add_option('2', 'Enter existing datetime frame from the database (somewhat lazy method)', '2')
        p.add_option('3', 'Enter a datetime frame and power usage manually (accurate method)', '3')
        option = p.choose_option()

        _datetime, power_usage, appliance_in_use = [], [], []

        if option == '1':
            _datetime, power_usage, appliance_in_use = self.gather_interesting_data()
        elif option == '2':
            _datetime, power_usage, appliance_in_use = self.gather_data_from_given_data(p)
        elif option == '3':
            _datetime, power_usage, appliance_in_use = self.gather_manually_entered_data(p)
        else:
            p.print_line('Invalid option. Please try again.')
            self.screen(p)

        predictions, probabilities = self.predict_model.predict(_datetime, power_usage, appliance_in_use)

        self.predict_model.visualize(predictions, appliance_in_use, probabilities, show_plot=True)

        p.request_input('Press enter to continue...')
        p.to_previous_screen()

    def gather_interesting_data(self):
        return self.document_model.get_interesting_datetime(self.predict_model.appliance)

    def gather_data_from_given_data(self, p):
        while True:
            chosen_datetime = p.request_input('Enter a datetime existing in the database (YYYY-MM-DD HH:MM:SS): ')

            get_real_datetime = Database.query(Query.SELECT_WHERE.format('PowerUsage', 'datetime', chosen_datetime))

            if len(get_real_datetime) > 0:
                return self.predict_model.use_existing_data(chosen_datetime)
            else:
                p.print_line('Datetime not found in the database. Please try again.')

    def gather_manually_entered_data(self, p):
        datetimes = []
        power_usages = []
        appliances_in_use = []

        while True:
            while True:
                chosen_datetime = p.request_input('Enter a datetime (YYYY-MM-DD HH:MM:SS): ')
                # check if the chosen datetime is valid
                if not chosen_datetime:
                    p.print_line('Enter a valid datetime.')
                    continue
                else:
                    try:
                        test_datetime = datetime.datetime.strptime(chosen_datetime, "%Y-%m-%d %H:%M:%S")
                        datetimes.append(chosen_datetime)
                        break
                    except ValueError:
                        p.print_line('Enter a valid datetime.')
                        continue
            while True:
                power_usage = p.request_input('Enter the power usage in watts: ')
                # check if the power usage is a valid integer
                if power_usage.isdigit():
                    power_usages.append(int(power_usage))
                    break
                else:
                    p.print_line('Enter a valid power usage.')
                    continue
            while True:

                appliance_in_use = p.request_input('Is the appliance in use? (1/0): ')
                if appliance_in_use in ['1', '0']:
                    appliances_in_use.append(bool(int(appliance_in_use)))
                    break
                else:
                    p.print_line('Enter a valid input.')
                    continue

            if p.request_input('Do you want to enter another datetime and power usage? (Y/N) [Y]: ').lower() == 'n':
                break

        return datetimes, power_usages, appliances_in_use

