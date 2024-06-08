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
        applianceID = 6
        power_usage = self.db.query("SELECT * FROM PowerUsage")
        appliance_in_use = self.db.query(f"SELECT * FROM IsUsingAppliance WHERE Appliance_id = {applianceID}")

        # Create dataframes
        power_usage_df = pd.DataFrame(power_usage)
        appliance_in_use_df = pd.DataFrame(appliance_in_use)

        power_usage_df['appliance_in_use'] = 0

        # Create a set of PowerUsage_id values that exist in appliance_in_use_df
        appliance_ids_set = set(appliance_in_use_df['PowerUsage_id'])

        # Update 'appliance_in_use' to 1 if id is in the set
        power_usage_df['appliance_in_use'] = power_usage_df['id'].apply(lambda x: 1 if x in appliance_ids_set else 0)

        # Extract timestamp information
        power_usage_df['datetime'] = pd.to_datetime(power_usage_df['datetime'])
        power_usage_df['weekday'] = power_usage_df['datetime'].dt.dayofweek
        power_usage_df['hour'] = power_usage_df['datetime'].dt.hour
        power_usage_df['minute'] = power_usage_df['datetime'].dt.minute

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
        os.makedirs(f'data/{applianceID}', exist_ok=True)

        # Save to CSV files
        train_data.to_csv(f'data/{applianceID}/train_data.csv', index=False)
        test_data.to_csv(f'data/{applianceID}/test_data.csv', index=False)

        p.request_input("Press enter to continue: ")

        p.to_previous_screen()