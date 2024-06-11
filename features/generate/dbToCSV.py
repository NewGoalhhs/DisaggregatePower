from core.Generate import Generate
from core.Database import Database
from helper.LoadingBarHelper import LoadingBarHelper

import pandas as pd
import os

class dbToCSV(Generate):
    def __init__(self):
        super().__init__()
        self.db = Database()

    def run(self, p):
        # Get all data from the database
        power_usage = self.db.query("SELECT * FROM PowerUsage")
        appliances_in_use = self.db.query("SELECT * FROM IsUsingAppliance")

        # Create dataframes
        power_usage_df = pd.DataFrame(power_usage)
        appliances_in_use_df = pd.DataFrame(appliances_in_use)

        # Create a dictionary with PowerUsage_id as keys and list of Appliance_id as values
        appliance_usage_dict = appliances_in_use_df.groupby('PowerUsage_id')['Appliance_id'].apply(list).to_dict()

        # Add a new column 'appliances_in_use' with the list of appliances
        power_usage_df['appliances_in_use'] = power_usage_df['id'].apply(lambda x: appliance_usage_dict.get(x, []))

        # Extract timestamp information
        power_usage_df['datetime'] = pd.to_datetime(power_usage_df['datetime'])
        power_usage_df['weekday'] = power_usage_df['datetime'].dt.dayofweek
        power_usage_df['hour'] = power_usage_df['datetime'].dt.hour

        # Round the minute to the nearest 15 minutes interval
        power_usage_df['minute'] = power_usage_df['datetime'].dt.minute
        power_usage_df['minute'] = power_usage_df['minute'].apply(lambda x: 15 * round(x/15))

        # Split the data into weekdays and weekends
        weekday_data = power_usage_df[power_usage_df['weekday'] < 5]  # 0-4 are weekdays
        weekend_data = power_usage_df[power_usage_df['weekday'] >= 5]  # 5-6 are weekends

        # Split weekdays data
        weekday_train = weekday_data.sample(frac=0.8, random_state=42)
        weekday_test = weekday_data.drop(weekday_train.index)

        # Split weekends data
        weekend_train = weekend_data.sample(frac=0.8, random_state=42)
        weekend_test = weekend_data.drop(weekend_train.index)

        # Combine training and testing datasets
        train_data = pd.concat([weekday_train, weekend_train]).reset_index(drop=True)
        test_data = pd.concat([weekday_test, weekend_test]).reset_index(drop=True)

        # Create directories
        os.makedirs('data/multiclass', exist_ok=True)

        # Save to CSV files
        train_data.to_csv('data/multiclass/train_data.csv', index=False)
        test_data.to_csv('data/multiclass/test_data.csv', index=False)

        p.request_input("Press enter to continue: ")

        p.to_previous_screen()