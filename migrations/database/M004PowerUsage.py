from core.Migration import Migration
from SQL.SQLQueries import PowerUsageOperations as Query
from SQL.SQLQueries import BuildingOperations as BuildingQuery
from core.Database import Database
import pandas as pd


class M004PowerUsage(Migration):
    def up(self):
        self.add_sql(Query.CREATE_POWER_USAGE_TABLE)
        self.add_sql(Query.CREATE_POWER_USAGE_INDEX)

    def down(self):
        self.add_sql(Query.DROP_POWER_USAGE_TABLE)

    def insert(self, csv_path):
        data_frames = []
        if '_' in csv_path:
            reddus = True
            building = ("REDDUS_" + csv_path.split('_')[1].split('.')[0])
        else:
            reddus = False
            building = ("DATA_" + csv_path.split('-')[0].split('r')[1])
        building_id = Database.query(BuildingQuery.GET_BUILDING_ID.format(building))[0].get('id')

        if reddus:
            df = pd.read_csv(csv_path).dropna()
        else:
            df = pd.read_csv(csv_path, delimiter=';').dropna()

        if reddus:
            df['time'] = pd.to_datetime(df['time'])
        else:
            current_time = pd.Timestamp('2020-01-01')
            # each period is 15 minutes
            for index, row in df.iterrows():
                df.at[index, 'time'] = current_time
                current_time += pd.Timedelta(15, unit='m')
            df = df.drop(columns=['Periods'])

        df['building_id'] = building_id

        if 'main' not in df.columns:
            df['Total Consumption'] = df['Total Consumption'].str.replace(',', '.').astype(float)
            df['Total Consumption'] = df['Total Consumption'].apply(lambda x: x * 100)
            df['main'] = df['Total Consumption']
        data_frames.append(df[['building_id', 'time', 'main']])

        final_df = pd.concat(data_frames, ignore_index=True)

        self.set_loading_bar_status('Retrieving data')
        self.set_loading_bar_goal(len(final_df)*2)

        for x in final_df.to_numpy():
            self.lb.update(1)
            record = tuple(x)
            self.add_sql(Query.INSERT_POWER_USAGE.format(*record))